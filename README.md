![image](https://github.com/patternizer/glosat-optimal-average/blob/main/glosat-v-lek-lsat-australia-cost_analysis-bestfit.png)
![image](https://github.com/patternizer/glosat-optimal-average/blob/main/glosat-v-lek-lsat-india-cost_analysis-bestfit.png)

# glosat-optimal-average

Algorithm to detect a minimal sample of long station timeseries for reconstruction of the global mean. Part of ongoing work for the [GloSAT Project](https://www.glosat.org):

## Contents

* `optimal_average.py` - main python code
* `ensemble_func.py` - auxilliary code containing PCA functions

The first step is to clone the latest glosat-optimal-avergae code and step into the check out directory: 

    $ git clone https://github.com/patternizer/glosat-optimal-average.git
    $ cd glosat-optimal-average

### Using Standard Python

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested 
in a conda virtual environment running a 64-bit version of Python 3.8.11+.

glosat-optimal-average scripts can be run from sources directly once the global pickled pandas dataframe global archive file df_anom.pkl is placed in the DATA/ directory.  

Run with:

    $ python optimal_average.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)






