<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
<!-- [![MIT License][license-shield]][license-url] -->
<!--[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!--<a href="https://github.com/Gotticketsto360tour/LaDoN">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">Tie-deletion can facilitate Cooperation</h3>

  <p align="center">
    The underlying code for the LaDoN project
    <br />
    <a href="https://github.com/Gotticketsto360tour/LaDoN"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/Gotticketsto360tour/LaDoN/issues">Report Bug</a>
    ·
    <a href="https://github.com/Gotticketsto360tour/LaDoN/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This repository provides the underlying code for the thesis "Tie-deletion can facilitate Cooperation". All the code is made to be as reproducible as possible. To reproduce the results, follow the steps provided bellow.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With
* [Networkx](https://networkx.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Optuna](https://optuna.org/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

The code is written in Python with popular libraries. As is always the case, it is recommended to setup a virtual environment for the project to avoid dependency issues. There are a number of good alternatives for creating such virtual environments. This repo was made using [Anaconda](https://www.anaconda.com/python-r-distribution?utm_campaign=python&utm_medium=online-advertising&utm_source=google&utm_content=anaconda-download&gclid=CjwKCAjwyryUBhBSEiwAGN5OCArsHSYImwyYZrXmaMCYQcC4Y3eZtWk7DVazjwohIjHVXORLDGh_4hoCZdgQAvD_BwE), but other virtual environments will work just as well. 

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Gotticketsto360tour/LaDoN.git
   ```
2. Install the dependencies. Both a `requirements.txt` and a `environment.yml` is provided. Using the `environment.yml` can be done with
    ```sh
   conda env create -f environment.yml
   ```

    Using the `requirements.txt` can be done with
    ```sh
   conda create -n ladon python=3.9
   conda activate ladon
   pip install -r requirements.txt
    ```
3. Install the `ladon`-package
   ```python
   python setup.py install
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Mikkel Werling - [@twitter_handle](https://twitter.com/twitter_handle) - werling1407@gmail.com

Project Link: [https://github.com/Gotticketsto360tour/LaDoN](https://github.com/Gotticketsto360tour/LaDoN)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Riccardo Fusaroli](https://github.com/fusaroli)
* [Paul Smaldino](https://github.com/psmaldino)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Gotticketsto360tour/LaDoN.svg?style=for-the-badge
[contributors-url]: https://github.com/Gotticketsto360tour/LaDoN/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Gotticketsto360tour/LaDoN.svg?style=for-the-badge
[forks-url]: https://github.com/Gotticketsto360tour/LaDoN/network/members
[stars-shield]: https://img.shields.io/github/stars/Gotticketsto360tour/LaDoN.svg?style=for-the-badge
[stars-url]: https://github.com/Gotticketsto360tour/LaDoN/stargazers
[issues-shield]: https://img.shields.io/github/issues/Gotticketsto360tour/LaDoN.svg?style=for-the-badge
[issues-url]: https://github.com/Gotticketsto360tour/LaDoN/issues
[license-shield]: https://img.shields.io/github/license/Gotticketsto360tour/LaDoN.svg?style=for-the-badge
[license-url]: https://github.com/Gotticketsto360tour/LaDoN/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png