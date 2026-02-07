from pygments.lexers.sql import TransactSqlLexer
from triton.language import trans
import sys
from fontTools.t1Lib import write
from typing import Any
from charset_normalizer import from_bytes
import numpy as np
import struct
import cv2
import os
import pickle

import lmdb
from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray


class DataStore(metaclass=ABCMeta):
    """
    A storage layer with different backends
    """
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, '__init__') and callable(subclass.__init__) and
            hasattr(subclass, '__enter__') and callable(subclass.__enter__) and
            hasattr(subclass, '__exit__') and callable(subclass.__exit__) and
            hasattr(subclass, '__len__') and callable(subclass.__len__) and
            hasattr(subclass, 'get') and callable(subclass.get) and
            hasattr(subclass, 'append') and callable(subclass.append) and
            hasattr(subclass, 'set_metadata') and callable(subclass.set_metadata) and
            hasattr(subclass, 'get_metadata') and callable(subclass.get_metadata)            
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
    def set_metadata(self, key: str, value: int | float | str):
        pass
    
    @abstractmethod
    def get_metadata(self, key: str, dtype: type) -> Any:
        pass



class LMDBStore(DataStore):
    def __init__(self, path: str, write: bool):
        self._path = path
        self.path_exists = os.path.exists(self._path)
        self._write = write
        
        self.len = 0
        self.metadata = dict()
        self.metadata['size'] = None
        
        self.env = None
        self.transaction = None
        self.env_pid = None
        self.commitFrequency = 100 if write else None
                
        
    def __enter__(self):
        if self._write and self.metadata['size'] is None and not self.path_exists:
            raise ValueError("To create a new store you need to set a size in metadata")
        self.env = self._get_or_init_env()
        self.transaction = self._get_or_init_transaction()
        
        length_bytes = self.transaction.get('__len__'.encode('ascii'))
        if length_bytes:
            self.len = int.from_bytes(length_bytes, 'big')
        else:
            self.len = 0
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.transaction:
            if exc_type:
                print(f'Aborting due to error: {exc_value}', file=sys.stderr)
                self.transaction.abort()
            else:
                try:
                    if self._write:
                        self.transaction.put('__len__'.encode('ascii'), self.len.to_bytes(64, 'big'))
                        self.transaction.commit()
                except lmdb.Error as e:
                    print(f'Error during transaction commit: {e}')
                    self.transaction.abort()
        if self.env and self._write:
            self.env.sync()
        self.env.close()
        
    def set_metadata(self, key: str, value: int | float | str):
        self.metadata[key] = value
        
    def get_metadata(self, key: str, dtype: type):
        return self.metadata.get(key)
        
    def get(self, idx: int) -> tuple[NDArray, NDArray]:
        if idx < 0 or idx >= self.len:
            raise IndexError(f"Index {idx} out of bounds for length {self.len}")
        
        ikey = f'i{idx}'.encode('ascii')
        lkey = f'l{idx}'.encode('ascii')
        self.transaction = self._get_or_init_transaction()
        img = self.transaction.get(ikey)
        lab = self.transaction.get(lkey)
        if img is None: raise ValueError(f'Image {id} not found in the store')
        if lab is None: raise ValueError(f'Label {id} not found in the store')
        return pickle.loads(img), pickle.loads(lab)
        
    def _get_or_init_transaction(self):
        if self.transaction is None:
            self.env = self._get_or_init_env()
            self.transaction = self.env.begin(write=self._write)
        return self.transaction
        

    def _get_or_init_env(self):
        if self.env is None or (self.env_pid and self.env_pid != os.getpid()):
            target_size = self.metadata.get('size')

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
                subdir=True
            )
            self.env_pid = os.getpid()
        return self.env
    

    def __len__(self):
        return self.len

    def path(self) -> str:
        return self._path

    def append(self, image: NDArray, label: NDArray):
        if not self._write:
            ValueError("Store cannot write in read-only mode.")
        img = pickle.dumps(image)
        lab = pickle.dumps(label)
        count = self.len
        self.transaction = self._get_or_init_transaction()
        self.transaction.put(f"i{count}".encode('ascii'), img)
        self.transaction.put(f"l{count}".encode('ascii'), lab)
        if self.commitFrequency and self.commitFrequency % count == 0:
            self.transaction.commit()
        self.len += 1
        
    
    def close(self):
        self.__exit__(None, None, None)





def new_lmdb_store(path: str, write: bool) -> LMDBStore:
    return LMDBStore(path, write)


if __name__ == '__main__':
    # store = new_lmdb_store('test.lmdb', write=True, metadata={'h': 100, 'w': 200, 'len:': 1})
    # store.put('example', 3)
    # store.close()
    
    # reopen = new_lmdb_store('test.lmdb', write=False, metadata=None)
    # print(reopen.get('example', int))
    # reopen.close()
    
    write_store = new_lmdb_store('./test.lmdb', write=True)
    write_store.set_metadata('corners_recess_percentage', 0.0)
    write_store.set_metadata('size', 100 * 1024 * 1024)
        
    with write_store as store:
        img = cv2.imread('/home/antonio/Downloads/smartdoc15/extended_smartdoc_dataset/validation/datasheet/0_in.png', cv2.IMREAD_COLOR_BGR)
        if img is None:
            print(f'Couldn\' load image...', file=sys.stderr)
            exit(2)
        store.append(img, np.array([1., 2., 3.], dtype=float))
        print(f'Stored an image of shape {img.shape}')
        print(f'Store size: {len(store)}')
    
    with new_lmdb_store('test.lmdb', write=False) as store:
        print(len(store))
        img, _ = store.get(0)
        print(f'Retrieved an image of shape {img.shape}')
        cv2.imwrite('test.jpg', img)
        
    