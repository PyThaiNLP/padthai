from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8-sig") as f:
    readme = f.read()

with open("requirements.txt", "r", encoding="utf-8-sig") as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
    name="padthai",
    version="0.2.0",
    description="Make Pad Thai From few-shot learning 😉",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Wannaphong Phatthiyaphaibun",
    author_email="wannaphong@yahoo.com",
    packages=find_packages(exclude=["tests", "tests.*"]),
    url="https://github.com/PyThaiNLP/padthai",
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    project_urls={
        "Documentation": "https://pythainlp.github.io/padthai/",
        "Tutorials": "https://github.com/PyThaiNLP/padthai/tree/main/notebooks",
        "Source Code": "https://github.com/PyThaiNLP/padthai",
        "Bug Tracker": "https://github.com/PyThaiNLP/padthai/issues",
    }
)
