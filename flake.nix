{
  description = "c/c++ development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self , nixpkgs ,... }: let
    system = "x86_64-linux";
  in {
    devShells."${system}".default = let
      pkgs = import nixpkgs {
        inherit system;
      };
    in pkgs.mkShell {
      packages = with pkgs; [
        cmake
        rocmPackages.clr
        rocmPackages.rocm-device-libs
        glfw
        vulkan-headers
        vulkan-loader
        vulkan-tools
        vulkan-validation-layers
        glslang
        valgrind
      ];
      
      shellHook = ''
        git submodule update --init --recursive
        export PATH="${pkgs.rocmPackages.clr}/bin:$PATH"
        export HIP_DEVICE_LIB_PATH=${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode
        export LD_LIBRARY_PATH=${pkgs.rocmPackages.clr}/lib:$LD_LIBRARY_PATH
      '';
    };
  };
}
