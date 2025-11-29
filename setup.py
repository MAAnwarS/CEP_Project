"""
What is setup.py?
1. It is a script used for packaging and distributing a Python project.
2. It tells tools like pip how to install your project, its metadata (name, version, author, etc.), and what dependencies it needs.
"""


from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:

    requirement_lst = []

    try:
        with open('requirements.txt', 'r') as file:
            # Read lines from the file
            lines = file.readlines()
            # Process each line
            for line in lines:
                requirement = line.strip()
                # Ignore empty lines and .e
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)

    except FileNotFoundError:
        print('File -> requirements.txt not found')
    
    return requirement_lst


setup(
    name = 'NetworkSecurity',
    version = '0.0.1',
    author = 'Ammar Vohra',
    author_email = 'ammarvohra92@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements()
)