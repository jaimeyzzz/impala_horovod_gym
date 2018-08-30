If you happen to be from internal Tencent and have access to the c.oa.com "compute sharing platform",
here is a description of how to prepare the CSV file needed by `run_experiment_mm_raw.py`.

After your application has been finalized, you should be able to download a CSV file
that describes all the machines you've successfully applied from c.oa.com. 
Let's call it `a.csv`.
We then provide the script `tmp_parse_coa.py` that reads in `a.csv` 
and outputs the CSV (call it `b.csv`) that is needed by `run_experiment_mm_raw.py`.  

The workflow should go similar when using other cloud service.

