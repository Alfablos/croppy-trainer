import shutil
import sys

from pathlib import Path
from typing import Any
import numpy as np
import os
import pyarrow as pa

import lmdb
from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray

from common import DEFAULT_STORAGE_CLASS


class DataStore(metaclass=ABCMeta):
    """
    A storage layer with different backends
    """

    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__init__")
            and callable(subclass.__init__)
            and hasattr(subclass, "__enter__")
            and callable(subclass.__enter__)
            and hasattr(subclass, "__exit__")
            and callable(subclass.__exit__)
            and hasattr(subclass, "__len__")
            and callable(subclass.__len__)
            and hasattr(subclass, "get")
            and callable(subclass.get)
            and hasattr(subclass, "append")
            and callable(subclass.append)
            and hasattr(subclass, "set_metadata")
            and callable(subclass.set_metadata)
            and hasattr(subclass, "get_metadata")
            and callable(subclass.get_metadata)
            and hasattr(subclass, "compact")
            and callable(subclass.compact)
            or NotImplemented
        )

    @abstractmethod
    def __init__(self, path: str, write: bool):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get(self, idx: int) -> tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def append(self, image: NDArray, label: NDArray):
        pass

    @abstractmethod
    def set_metadata(self, key: str, value: str):
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Any:
        pass

    @abstractmethod
    def compact(self, dst_path: str):
        pass

    @abstractmethod
    def storage_class(self) -> str:
        pass


class LMDBStore(DataStore):
    def __init__(self, path: str, write: bool):
        self._path = path
        self.path_exists = os.path.exists(self._path)
        self._write = write

        self.len = 0
        self.metadata = dict()
        self.metadata["size"] = None

        self.env = None
        self.transaction = None
        self.img_size_set = False
        self.env_pid = None
        self.commitFrequency = 100 if write else None

    def __enter__(self):
        if self._write and self.metadata["size"] is None and not self.path_exists:
            raise ValueError("To create a new store you need to set a size in metadata")

        if self._write and (
            self.metadata.get("h") is None or self.metadata.get("w") is None
        ):
            raise ValueError(
                'Refusing to open LMDB store: metadata must contain "h" and "w", set them with `set_metadata(...)` and try again.'
            )

        self.env = self._get_or_init_env()
        self.transaction = self._get_or_init_transaction()

        length_bytes = self.transaction.get("__len__".encode("utf-8"))
        if length_bytes:
            self.len = int.from_bytes(length_bytes, "big")
        else:
            self.len = 0

        h = self.transaction.get("h".encode("utf-8"))
        if h:
            self.metadata["h"] = int(h)
        w = self.transaction.get("w".encode("utf-8"))
        if h:
            self.metadata["w"] = int(w)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.transaction:
            if exc_type:
                print(f"Aborting due to error: {exc_value}", file=sys.stderr)
                self.transaction.abort()
            else:
                try:
                    if self._write:
                        self.transaction.put(
                            "__len__".encode("utf-8"), self.len.to_bytes(64, "big")
                        )
                        self.transaction.commit()
                except lmdb.Error as e:
                    print(f"Error during transaction commit: {e}")
                    self.transaction.abort()
        if self.env and self._write:
            self.env.sync()
        self.env.close()

    def set_metadata(self, key: str, value: int | float | str):
        self.metadata[key] = value

    def get_metadata(self, key: str):
        return self.metadata.get(key)

    def get(self, idx: int) -> tuple[NDArray, NDArray]:
        if idx < 0 or idx >= self.len:
            raise IndexError(f"Index {idx} out of bounds for length {self.len}")

        ikey = f"i{idx}".encode("utf-8")
        lkey = f"l{idx}".encode("utf-8")
        self.transaction = self._get_or_init_transaction()
        img = self.transaction.get(ikey)
        lab = self.transaction.get(lkey)
        if img is None:
            raise ValueError(f"Image {idx} not found in the store")
        if lab is None:
            raise ValueError(f"Label {idx} not found in the store")
        image = np.frombuffer(img, dtype=np.uint8).reshape(
            self.metadata["h"], self.metadata["w"], 3
        )
        label = np.frombuffer(lab, dtype=np.float32).reshape(4, 2)
        return image, label

    def _get_or_init_transaction(self):
        if self.transaction is None:
            self.env = self._get_or_init_env()
            self.transaction = self.env.begin(write=self._write)
        return self.transaction

    def _get_or_init_env(self):
        if self.env is None or (self.env_pid and self.env_pid != os.getpid()):
            target_size = self.metadata.get("size")

            if target_size is not None:
                # If the user specifies the size
                map_size = target_size
            elif self.path_exists and not self._write:
                # read-only mode
                map_size = 0
            else:
                # Open with max virtual size
                is_64bit = sys.maxsize > 2**32
                map_size = 1099511627776 if is_64bit else 104857600

            self.env = lmdb.open(
                self._path,
                map_size=map_size,
                readonly=not self._write,
                lock=self._write,
                readahead=False,
                meminit=False,
                subdir=True,
            )
            self.env_pid = os.getpid()
        return self.env

    def __len__(self):
        return self.len

    def path(self) -> str:
        return self._path

    def append(self, image: NDArray, label: NDArray):
        if not self._write:
            raise ValueError("Store cannot write in read-only mode.")
        img = image.astype(np.uint8).tobytes()
        lab = label.astype(np.float32).tobytes()
        count = self.len
        self.transaction = self._get_or_init_transaction()

        self.transaction.put(f"i{count}".encode("utf-8"), img)
        self.transaction.put(f"l{count}".encode("utf-8"), lab)
        if self.commitFrequency and count > 0 and count % self.commitFrequency == 0:
            self.transaction.commit()
            self.transaction = self.env.begin(write=self._write)
        self.len += 1

    def compact(self, dst_path: str):
        if not self._write:
            raise ValueError("Store cannot compact in read-only mode.")

        # Ensure metadata and data are committed before copying
        self.transaction = self._get_or_init_transaction()
        self.transaction.put("__len__".encode("utf-8"), str(self.len).encode("utf-8"))
        self.transaction.put(
            "h".encode("utf-8"), str(self.metadata["h"]).encode("utf-8")
        )
        self.transaction.put(
            "w".encode("utf-8"), str(self.metadata["w"]).encode("utf-8")
        )
        self.transaction.commit()
        self.env.copy(dst_path, compact=True)
        self.transaction = self.env.begin(write=self._write)

    def close(self):
        self.__exit__(None, None, None)

    def storage_class(self):
        return "lmdb"


class ArrowStore(DataStore):
    def __init__(self, path: str, write: bool):
        self._path = path
        self.path_exists = os.path.exists(self._path)
        self._write = write
        self.schema = pa.schema(
            fields=[
                pa.field("image", pa.binary()),
                # labels, from arrow's perspective are just bytes against which no query should be executed
                # this unifies label handling (`.tobytes()`) between resnet and unet
                pa.field("label", pa.binary()),
            ]
        )

        self.len = 0
        self.metadata = dict()
        self.img_size_set = False

        self.image_buffer = []
        self.label_buffer = []
        self.batch_size = 100

        self.writer = None
        self.reader = None
        self.table = None

    def __enter__(self):
        if self._write:
            if self.path_exists:
                raise FileExistsError(
                    f"The arrow store enforces immutability and data exists at {self._path}. Refusing to continue."
                )

            meta = {
                k.encode("utf-8"): v.encode("utf-8") for k, v in self.metadata.items()
            }
            self.schema = self.schema.with_metadata(meta)
            self.sink = pa.OSFile(self._path, "wb")
            self.writer = pa.ipc.new_file(
                sink=self.sink, schema=self.schema, metadata=meta
            )
        else:
            if not self.path_exists:
                raise FileNotFoundError(
                    f"Cannot create arrow store from file at {self._path}: file not found."
                )

            source = pa.memory_map(self._path, "r")
            self.reader = pa.ipc.open_file(source)
            self.table = self.reader.read_all()

            if self.table.schema.metadata:
                self.metadata = {
                    k.decode("utf-8"): v.decode("utf-8")
                    for k, v in self.table.schema.metadata.items()
                }

        if self.metadata.get("h") is None or self.metadata.get("w") is None:
            raise ValueError(
                'Refusing to open arrow store: metadata must contain "h" and "w", set them with `set_metadata(...)` and try again.'
            )

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._write:
            if self.image_buffer:
                self._flush()

            if self.writer:
                self.writer.close()
            if self.sink:
                self.sink.close()
        else:
            # self.reader.close()
            pass

    def __len__(self):
        if self._write:
            return self.len
        else:
            return self.table.num_rows if self.table else 0

    def get(self, idx: int) -> tuple[NDArray, NDArray]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Invalid index `{idx}` for store of length {len(self)}.")

        ibuf = self.table["image"][idx].as_py()
        lbuf = self.table["label"][idx].as_py()

        image = np.frombuffer(ibuf, dtype=np.uint8).reshape(
            int(self.metadata["h"]), int(self.metadata["w"]), 3
        )
        label = np.frombuffer(lbuf, dtype=np.float32).reshape(4, 2)

        return image, label

    def append(self, image: NDArray, label: NDArray):
        if not self._write:
            raise ValueError("Store cannot write in read-only mode.")
        # turns (H, W, 3) into (1,) shape (faster than pickle)
        img = image.astype(np.uint8).tobytes()
        # TODO: accomodate for mask labels, not just coordinates. Masks are same h x w but B/W (1 channel)
        lab = label.astype(np.float32).tobytes()
        self.image_buffer.append(img)
        self.label_buffer.append(lab)

        if len(self.image_buffer) >= self.batch_size:
            self._flush()

        self.len += 1

    def set_metadata(self, key: str, value: str):
        self.metadata[key] = value

    def get_metadata(self, key: str):
        return self.metadata.get(key)

    def _flush(self):
        if not self.image_buffer:
            return

        images = pa.array(self.image_buffer, type=pa.binary())
        labels = pa.array(self.label_buffer, type=pa.binary())

        batch = pa.RecordBatch.from_arrays(arrays=[images, labels], schema=self.schema)

        self.writer.write_batch(batch)
        self.image_buffer.clear()
        self.label_buffer.clear()

    def compact(self, dst_path: str):
        shutil.copytree(self._path, dst_path, symlinks=True, dirs_exist_ok=False)

    def storage_class(self) -> str:
        return "arrow"


def merge_arrow_stores(input_paths: list[str], output_path: str):
    """
    Merge multiple Arrow IPC stores into a single one.
    All input stores must have the same dimensions (h, w).
    """
    if os.path.exists(output_path):
        raise FileExistsError(f"Output path {output_path} already exists.")

    tables: list[pa.Table] = []
    metadata: dict[str, str] | None = None
    total_rows = 0

    for path in input_paths:
        source = pa.memory_map(path, "r")
        reader = pa.ipc.open_file(source)
        table = reader.read_all()
        rows = table.num_rows
        total_rows += rows

        if table.schema.metadata:
            file_meta = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
            if metadata is None:
                metadata = file_meta
            else:
                if file_meta.get("h") != metadata.get("h") or file_meta.get("w") != metadata.get("w"):
                    raise ValueError(
                        f"Dimension mismatch: first store is {metadata['h']}x{metadata['w']}, "
                        f"but {path} is {file_meta['h']}x{file_meta['w']}"
                    )

        tables.append(table)
        print(f"  {path}: {rows} rows")

    merged = pa.concat_tables(tables)

    # Build schema with metadata
    schema = merged.schema
    if metadata:
        meta_bytes = {k.encode(): v.encode() for k, v in metadata.items()}
        schema = schema.with_metadata(meta_bytes)

    # Write merged table as IPC file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sink = pa.OSFile(output_path, "wb")
    writer = pa.ipc.new_file(sink=sink, schema=schema)

    for batch in merged.to_batches(max_chunksize=100):
        writer.write_batch(batch)

    writer.close()
    sink.close()

    print(f"Merged {len(tables)} stores → {output_path} ({total_rows} total rows)")


def new_store(path: str, write: bool) -> DataStore:
    suffix = Path(path).suffix
    suffix_lower = suffix.lower()
    if suffix_lower == ".arrow":
        return ArrowStore(path=path, write=write)
    elif suffix_lower == ".lmdb":
        return LMDBStore(path, write)
    else:
        raise NotImplementedError(
            f"No storage classes configured to handle `{suffix}` files."
        )


if __name__ == "__main__":
    # store = new_lmdb_store('test.lmdb', write=True, metadata={'h': 100, 'w': 200, 'len:': 1})
    # store.put('example', 3)
    # store.close()

    # reopen = new_lmdb_store('test.lmdb', write=False, metadata=None)
    # print(reopen.get('example', int))
    # reopen.close()

    # write_store = new_lmdb_store('./test.lmdb', write=True)
    # write_store.set_metadata('corners_recess_percentage', 0.0)
    # write_store.set_metadata('size', 100 * 1024 * 1024)
    #
    # with write_store as store:
    #     img = cv2.imread('/home/antonio/Downloads/smartdoc15/extended_smartdoc_dataset/validation/datasheet/0_in.png', cv2.IMREAD_COLOR_BGR)
    #     if img is None:
    #         print(f'Couldn\' load image...', file=sys.stderr)
    #         exit(2)
    #     store.append(img, np.array([1., 2., 3.], dtype=float))
    #     print(f'Stored an image of shape {img.shape}')
    #     print(f'Store size: {len(store)}')
    #
    # with new_lmdb_store('test.lmdb', write=False) as store:
    #     print(len(store))
    #     img, _ = store.get(0)
    #     print(f'Retrieved an image of shape {img.shape}')
    #     cv2.imwrite('test.jpg', img)

    store = ArrowStore("./arrow_test.arrow", write=True)
