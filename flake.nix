{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixpkgs-unstable";
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
            torch-bin
            torchvision-bin
            pillow
            opencv-python-headless
            matplotlib
            tqdm
            pynvml
          ]
          ++ (gpuDependantPackages pkgs)
        );

      dependencies = pkgs: with pkgs; [ ];

      mkLibraryPath =
        pkgs:
        with pkgs;
        lib.makeLibraryPath [
          stdenv.cc.cc # numpy (on which scenedetect depends) needs C libraries
          cudaPackages.cuda_nvrtc # libncrtc.so for cupy
          cudaPackages.cudatoolkit
          cudaPackages.libcutensor
          cudaPackages.libcublas
          cudaPackages.libcusolver
          cudaPackages.cuda_cudart
        ];

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
    in
    {
      packages = forAllSystems (pkgs: {
        default = self.packages.${pkgs.stdenv.hostPlatform.system}.croppy;
        croppy = pkgs.stdenv.mkDerivation {
          name = "mimic";
          propagatedBuildInputs = [ (pythonForPkgs pkgs) ];
          dontUnpack = true;
          installPhase = "install -Dm755 ${./croppy.py} $out/bin/croppy";
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
