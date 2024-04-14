import subprocess

def run_command(command, cwd=None):
    try:
        result = subprocess.run(command, check=True, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing command:", command)
        print(e.stderr)

def main():
    print("Updating bert.cpp submodule...")
    run_command("git submodule update --init --recursive")

    print("Updating ggml submodule...")
    run_command("git submodule update --init --recursive", cwd="bert.cpp")

    print("Downloading MiniLM-L6-v2 model...")
    run_command("pip3 install -r requirements.txt", cwd="bert.cpp")
    run_command("python3 models/download-ggml.py download all-MiniLM-L6-v2 q4_0", cwd="bert.cpp")

    print("Building bert.cpp...")
    build_dir = "bert.cpp/build"
    run_command("mkdir -p " + build_dir)
    run_command("cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release", cwd=build_dir)
    run_command("make", cwd=build_dir)

    print("Building glue.cpp...")
    run_command("make", cwd=".")

if __name__ == "__main__":
    main()
    