{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=13868c071cc73a5e9f610c47d7bb08e5da64fdd5";
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
      cudaLibraries = pkgs: with pkgs; [
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
      mkLibraryPath =
        pkgs:
        with pkgs;
        lib.makeLibraryPath [
          stdenv.cc.cc # numpy (on which scenedetect depends) needs C libraries
        ] ++ (cudaLibraries pkgs);

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
      packages = forAllSystems (pkgs:
      let
        python = pythonForPkgs pkgs;
      in  
      {
        default = self.packages.${pkgs.stdenv.hostPlatform.system}.croppy-trainer;
        croppy-trainer = pkgs.stdenv.mkDerivation {
          name = "croppy-trainer";
          src = fs.toSource {
            root = ./.;
            fileset = fs.unions [ (fs.fileFilter (file: file.hasExt "py") ./.) ];
          };
          nativeBuildInputs = [ pkgs.makeWrapper ];
          propagatedBuildInputs = [ python ] ++ (cudaLibraries pkgs);
          # makeWrapper creates an executable in $out/bin/croppy-trainer
          installPhase = ''
            mkdir -p $out/bin $out/libexec/croppy-trainer
            cp -r . $out/libexec/croppy-trainer
            makeWrapper ${python}/bin/python $out/bin/croppy-trainer \
              --add-flags "$out/libexec/croppy-trainer/main.py"
          '';
        };
      });
      apps = forAllSystems (pkgs: {
        default = self.apps.${pkgs.stdenv.hostPlatform.system}.main;
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
            inputsFrom = [ ];
            packages = [
              python
              pkgs.uv
              pkgs.ruff
            ]
            ++ (dependencies pkgs);

            shellHook = ''
              ${
                if cudaSupport then
                  ''
                    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${mkLibraryPath pkgs}:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
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
