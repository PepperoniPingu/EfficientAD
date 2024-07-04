{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  # make cuda work in non nixos systems by linking the cuda driver to where nix looks for hardware specific drivers:
  # > sudo ln -s /usr/lib64/libcuda.so.1 /run/opengl-driver/lib/libcuda.so.1

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs { 
        system = system;
	      config.allowUnfree = true;
      };
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.python311
            pkgs.python311Packages.pip
            pkgs.python311Packages.pandas
            pkgs.python311Packages.torch-bin
            pkgs.python311Packages.torchvision-bin
          ];

          shellHook = ''
            source .env
          '';
        };
      });
}

