{
  description = "Development environment for plast";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
      "https://cache.nixos-cuda.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "plast-env";

          packages = with pkgs; [
            cmake
            git
            pkg-config
            stdenv.cc
            openssl

            # Core CUDA toolkit dependencies needed by cudarc
            cudaPackages.cuda_nvcc # Compiler for custom .cu files
            cudaPackages.cuda_nvrtc # Runtime compilation engine for Ptx::from_src / from_file
            cudaPackages.cuda_cudart # CUDA Runtime bindings
            cudaPackages.libcublas # BLAS acceleration math layers

            # Rust Toolchain setup
            cargo
            clippy
            rustc
            rustfmt
            cargo-flamegraph
          ];

          shellHook = ''
            # Provide critical locate pointers for runtime NVRTC context assemblies
            export CUDA_PATH="${pkgs.cudaPackages.cuda_nvcc}"
            export CUDA_ROOT="${pkgs.cudaPackages.cuda_nvcc}"

            # Update linker tracking ranges so cudarc can easily load shared hardware object contexts
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudaPackages.cuda_nvcc}/lib:${pkgs.cudaPackages.cuda_nvrtc}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH"

            echo "=== Plast Dev Environment Active (Pure Rust & CUDA Toolchain Enabled) ==="
            echo "NVCC version: $(nvcc --version | grep release)"
          '';
        };
      }
    );
}
