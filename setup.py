import sys, os
from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    sys.exit(1)


ROOT_DIR = "deep_trees"

def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        print(path)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


def make_extension(ext_name):
    ext_path = ext_name.replace(".", os.path.sep)+".pyx"
    return Extension(
        ext_name,
        [ext_path],
        include_dirs = ["."],
        extra_compile_args = ["-O3", "-Wall"],
    )

# get the list of extensions
ext_names = scandir(ROOT_DIR)

print("Found extensions:", ext_names)

# and build up the set of Extension objects
extensions = [make_extension(name) for name in ext_names]

# finally, we can pass all this to distutils
setup(
    name="deep_trees",
    packages=["deep_trees"],
    ext_modules=extensions,
    cmdclass = {'build_ext': build_ext},
)
