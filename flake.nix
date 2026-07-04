{
  description = "A flake for Dolphin project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f (import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      }));
    in
    {
      devShells = forEachSystem (pkgs:
        let
          isLinux = pkgs.stdenv.isLinux;
          isDarwin = pkgs.stdenv.isDarwin;

          # Define the FHS environment only for Linux
          fhs = if isLinux then pkgs.buildFHSEnv {
            name = "dolphin-fhs-env";
            targetPkgs = pkgs: with pkgs; [
              python3
              python3Packages.pip
              python3Packages.virtualenv
              git
              git-lfs
              just
              pkg-config
              gcc
              gnumake

              # FHS core libraries for PyTorch, OpenCV, MuPDF, decord, etc.
              stdenv.cc.cc.lib
              glibc
              glib
              libGL
              libGLU
              zlib
              libffi
              openssl
              libxcrypt
              dbus
              fontconfig
              freetype
              util-linux
              
              # X11 & GUI libraries
              xorg.libX11
              xorg.libXext
              xorg.libXrender
              xorg.libXinerama
              xorg.libXcursor
              xorg.libXrandr
              xorg.libXi
              xorg.libXtst
              xorg.libXft
              xorg.libXdamage
              xorg.libXcomposite
              xorg.libXfixes
              xorg.libxcb
              xorg.xcbutil
              
              # Extra fonts and toolkit libs
              libthai
              fribidi
              harfbuzz
              pango
              cairo
              gdk-pixbuf
              gtk3
              alsa-lib
              nspr
              nss
              systemd
            ];
            profile = ''
              export IN_FHS_ENV=1
              # If .venv exists, automatically activate it
              if [ -d .venv ]; then
                source .venv/bin/activate
              fi
            '';
            runScript = "bash";
          } else null;

          # Shared development inputs
          sharedInputs = with pkgs; [
            just
            git
            python3
            python3Packages.pip
            python3Packages.virtualenv
          ] ++ pkgs.lib.optionals isDarwin (with pkgs; [
            libiconv
            zlib
          ]);
        in
        {
          default = pkgs.mkShell {
            name = "dolphin-shell";
            buildInputs = sharedInputs ++ pkgs.lib.optionals isLinux [ fhs ];

            shellHook = ''
              # Welcome message and instructions
              echo "=========================================================="
              echo " Dolphin Development Environment (Nix Shell)              "
              echo "=========================================================="
              echo "Available tools: python, pip, just"
              
              ${if isLinux then ''
                echo "System: Linux detected."
                echo "FHS Environment is configured."
                if [ -z "$DIRENV_IN_ENVRC" ] && [ -z "$IN_FHS_ENV" ]; then
                  echo "Entering FHS container..."
                  exec ${fhs}/bin/dolphin-fhs-env
                else
                  echo "Running inside FHS environment or via Direnv."
                fi
              '' else ''
                echo "System: Darwin (macOS) detected."
                echo "Activating development environment..."
                if [ -d .venv ]; then
                  source .venv/bin/activate
                  echo "Virtual environment activated."
                else
                  echo "Virtual environment (.venv) not found. Run 'just install' to initialize."
                fi
              ''}
              echo "=========================================================="
            '';
          };
        }
      );
    };
}