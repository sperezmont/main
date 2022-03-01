# yelmo_tools
*Libraries, modules and scripts to work with data from the model Yelmo (https://gmd.copernicus.org/articles/13/2805/2020/)*
* yelmo_functions.py        2021.12.20	General functions to calculate Yelmo outputs
* yelmo_plot_functions.py   2021.12.22	General functions to plot Yelmo outputs 
* yelmo_gif_functions.py    2022.01.13	General functions to make gifs with Yelmo outputs
* yelmo_plot_functions.py   2022.03.01  Script for plotting results (ABUMIP oriented)

# Quick guide
Clone `yelmo_tools`
```bash
git clone https://github.com/sperezmont/yelmo_tools.git
```
Edit lines 2, 3 and 4 of `config/config.sh` with your needs and configure the system
```bash
cd config/
chmod +x config.sh
config/config.sh
```
This will install the necessary Python dependencies that need the tools

Now, edit the interpreter lines (first line) in `yelmo_functions.py`, `yelmo_plot_functions.py` and `yelmo_gif_functions.py` with the interpreter you want. I recommend you to use the one that uses the virtual environment that we have created with `config.sh` to avoid problems
```python
#!/path/anaconda3/envs/env_name/bin/python_version
```

# Extra
You may also need to install LaTeX
```bash
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```
This will install also the needed packages for render the plots
