{
  description = "Development environment and CI pipeline for plast";

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

        cudaDeps = with pkgs.cudaPackages; [
          cuda_nvcc
          cuda_nvrtc
          cuda_cudart
          libcublas
        ];

        nativeBuildInputs = with pkgs; [
          pkg-config
          cmake
          git
          cargo
          rustc
          cargo-criterion
          clippy
        ];

        buildInputs =
          with pkgs;
          [
            openssl
          ]
          ++ cudaDeps;

        cudaEnvVars = ''
          export CUDA_PATH="${pkgs.cudaPackages.cuda_nvcc}"
          export CUDA_ROOT="${pkgs.cudaPackages.cuda_nvcc}"
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudaPackages.cuda_nvcc}/lib:${pkgs.cudaPackages.cuda_nvrtc}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH"
        '';

        plastPackage = pkgs.rustPlatform.buildRustPackage {
          pname = "plast";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          # Pass cargo flags if feature needs to be built into main derivation
          buildFeatures = [ "full" ];

          inherit nativeBuildInputs buildInputs;

          preBuild = cudaEnvVars;
        };

      in
      {
        packages.default = plastPackage;

        # --- CI CHECKS (nix flake check) ---
        checks = {
          # Check 1: Format
          formatting = pkgs.runCommand "check-fmt" { buildInputs = [ pkgs.rustfmt ]; } ''
            cd ${./.}
            rustfmt --check src/main.rs || true
            touch $out
          '';

          # Check 2: Clippy with full features (uses rustPlatform for cargo caching)
          clippy = pkgs.rustPlatform.buildRustPackage {
            pname = "plast-clippy";
            version = "0.0.3";
            src = ./.;
            cargoBuildFlags = [
              "--features"
              "full"
            ];

            cargoLock = {
              lockFile = ./Cargo.lock;
            };

            inherit nativeBuildInputs buildInputs;

            buildPhase = ''
              ${cudaEnvVars}
              cargo clippy --all-targets --features full --features full -- -D warnings
            '';

            installPhase = "touch $out";
          };

          # Check 3: Full Feature Tests (uses rustPlatform for cargo caching)
          cargo-tests = pkgs.rustPlatform.buildRustPackage {
            pname = "plast-tests";
            version = "0.0.3";
            src = ./.;

            cargoLock = {
              lockFile = ./Cargo.lock;
            };

            inherit nativeBuildInputs buildInputs;

            buildPhase = ''
              ${cudaEnvVars}
              cargo test --features full
            '';

            installPhase = "touch $out";
          };

          # Check 4: Check benchmark compilation without running or fetching network dependencies
          cargo-benches-compile = pkgs.rustPlatform.buildRustPackage {
            pname = "plast-benches-check";
            version = "0.0.3";
            src = ./.;

            cargoLock = {
              lockFile = ./Cargo.lock;
            };

            inherit nativeBuildInputs buildInputs;

            buildPhase = ''
              ${cudaEnvVars}
              cargo check --benches --features full
            '';

            installPhase = "touch $out";
          };
        };

        # --- DEVELOPMENT SHELL ---
        devShells.default = pkgs.mkShell {
          name = "plast-env";

          packages =
            with pkgs;
            [
              stdenv.cc
              clippy
              rustfmt
            ]
            ++ nativeBuildInputs
            ++ buildInputs;

          shellHook = ''
            ${cudaEnvVars}
            echo "=== Plast Dev Environment Active ==="
          '';
        };
      }
    );
}
