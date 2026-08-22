from setuptools import find_packages, setup

with open("README.md", "r", encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="rapidshot",
    version="2.4.0",
    description="A high-performance screencapture library for Windows using Desktop Duplication API",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zaatra/Rapidshot",
    author="Rapidshot Contributors",
    author_email="imzaatra@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Microsoft :: Windows :: Windows 8.1",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Graphics :: Capture",
        "Topic :: Multimedia :: Graphics :: Capture :: Screen Capture"
    ],
    install_requires=[
        "numpy>=1.19.0",
        "comtypes>=1.1.0",
    ],
    # Keep in step with [project.optional-dependencies] in pyproject.toml, which
    # is what actually builds the wheel; the reasoning for each entry is there.
    extras_require={
        "cv2": ["opencv-python>=4.5.0"],
        "gpu": ["cupy-cuda11x>=11.0.0"],
        "gpu_cuda12": ["cupy-cuda12x>=12.0.0"],
        "gpu_cuda13": ["cupy-cuda13x>=14.0.0"],
        "pil": ["pillow>=8.0.0"],
        "all": [
            "numpy>=1.19.0",
            "comtypes>=1.1.0",
            "opencv-python>=4.5.0",
            "pillow>=8.0.0"
        ]
    },
    python_requires=">=3.9",
)