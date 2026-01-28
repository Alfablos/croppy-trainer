{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=48698d12cc10555a4f3e3222d9c669b884a49dfe"; # nixpkgs-unstable";
  };
  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      pythonPackage = "python313";
      cudaSupport = true;

      forAllSystems =
        f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (
          system:
          f (
            import nixpkgs {
              inherit system;
              config.allowUnfree = true;
              config.cudaSupport = cudaSupport;
              config.cudaVersion = "12";
            }
          )
        );
      pythonForPkgs =
        pkgs:
        pkgs.${pythonPackage}.withPackages (
          pythonPackages:
          with pythonPackages;
          [
            numpy
            pandas
            pytest
            pyarrow
            torch-bin
            torchvision-bin
            tensorboard
            pillow
            opencv-python-headless
            matplotlib
            tqdm
            lmdb
          ]
          ++ (gpuDependantPackages pkgs)
        );

      dependencies = pkgs: with pkgs; [ ];
      cudaLibraries =
        pkgs: with pkgs; [
          cudaPackages.libcutensor
          cudaPackages.libcublas
          cudaPackages.libcusolver
          cudaPackages.libcufft
          cudaPackages.libcufile
          cudaPackages.libcurand
          cudaPackages.libcusparse
          cudaPackages.cuda_nvrtc # libncrtc.so for cupy
          cudaPackages.cudatoolkit
          cudaPackages.cuda_cudart
          cudaPackages.cuda_nvtx
          cudaPackages.cuda_cupti
          cudaPackages.cuda_nvrtc
          cudaPackages.cudnn
          # cudaPackages.cusparselt
          cudaPackages.nccl

        ];

      allLibrariesInPath =
        pkgs:
        with pkgs;
        [
          stdenv.cc.cc
        ]
        ++ (cudaLibraries pkgs);

      gpuDependantPackages =
        pkgs:
        with pkgs.${pythonPackage}.pkgs;
        if pkgs.config.cudaSupport then
          [ ]
          ++ (with pkgs; [
            cudatoolkit
            # libGLU
            # libGL
          ])
        else
          [ ];
      fs = nixpkgs.lib.fileset;
    in
    {
      packages = forAllSystems (
        pkgs:
        let
          python = pythonForPkgs pkgs;
        in
        {
          default = self.packages.${pkgs.stdenv.hostPlatform.system}.croppy-trainer;
          quickScript = pkgs.callPackage ./script.nix { };
          croppy-trainer = pkgs.stdenv.mkDerivation {
            name = "croppy-trainer";
            src = fs.toSource {
              root = ./trainer;
              fileset = fs.unions [ (fs.fileFilter (file: file.hasExt "py") ./trainer) ];
            };
            nativeBuildInputs = [ pkgs.makeWrapper ];
            propagatedBuildInputs = [
              python
            ]
            ++ (cudaLibraries pkgs);
            # makeWrapper creates an executable in $out/bin/croppy-trainer
            installPhase = ''
              mkdir -p $out/bin $out/libexec/croppy-trainer
              cp -r . $out/libexec/croppy-trainer
              makeWrapper ${python}/bin/python $out/bin/croppy-trainer \
                --add-flags "$out/libexec/croppy-trainer/croppy.py"
            '';
          };
        }
      );
      apps = forAllSystems (pkgs: {
        default = self.apps.${pkgs.stdenv.hostPlatform.system}.main;
        quickScript = {
          type = "app";
          program = "${self.packages.${pkgs.stdenv.hostPlatform.system}.quickScript}";
        };
        main = {
          type = "app";
          program = "${self.packages.${pkgs.stdenv.hostPlatform.system}.croppy-trainer}/bin/croppy-trainer";
        };
      });
      devShells = forAllSystems (pkgs: {
        default =
          let
            python = pythonForPkgs pkgs;
            cudaSupport = pkgs.config.cudaSupport;
          in
          pkgs.mkShell {
            inputsFrom = [ self.packages.${pkgs.stdenv.hostPlatform.system}.croppy-trainer ];
            packages = [
              python
              pkgs.uv
              pkgs.ruff
              pkgs.lmdb
            ]
            ++ (dependencies pkgs);

            shellHook = ''
              ${
                if cudaSupport then
                  ''
                    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath (allLibrariesInPath pkgs)}:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
                    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"
                    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                    export EXTRA_CCFLAGS="-I/usr/include"
                  ''
                else
                  ""
              }


              export PYTHONPATH="${python}/${python.sitePackages}"

              echo "=== PYTHON ==="
              echo
              echo "Setting PYTHONPATH to ${python}/${python.sitePackages}"
              export PYTHONPATH="${python}/${python.sitePackages}"
              echo Running $(python --version) @ $(which python) ${
                if pkgs.config.cudaSupport then "with CUDA support" else ""
              }
              echo

              # exec -l zsh
            '';
          };
      });
    };
}
