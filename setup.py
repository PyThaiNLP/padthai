from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8-sig") as f:
    readme = f.read()

with open("requirements.txt", "r", encoding="utf-8-sig") as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
    name="padthai",
    version="0.1-dev0",
    description="Make Pad Thai From few-shot learning 😉",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Wannaphong Phatthiyaphaibun",
    author_email="wannaphong@yahoo.com",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
)
