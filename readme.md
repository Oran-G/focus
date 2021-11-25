# REBASE data download and parse
### Download data from FTP
In Data repo:
> brew install inetutils
>ftp ftp.neb.com     (username: anonymous, password: your email address)
>cd pub/rebase       (to enter the correct directory)
>dir                 (to list the names of available files)
>get Readme |more    (to view the file called Readme, spaces matter!)
>get Readme          (to copy the file called Readme to your machine)
>quit                (returns you back to your local system).

On redhat:
```
module load ncftp/3.2.6
ncftp ftp.neb.com
cd pub/rebase
dir
get type2.110
get type2ref.110
get type2ref.txt
get type2.txt
get Type_II_methyltransferase_genes_DNA.txt
get Type_II_methyltransferase_genes_Protein.txt
get All_Type_II_restriction_enzyme_genes_Protein.txt
get All_Type_II_restriction_enzyme_genes_DNA.txt
exit
```

In code repo:
> pip install -r requirments.txt
> python3 parse.py
> create data paths in .yaml - must have all paths that are in oran.yaml
> python3 fasta.py
> python3 mmseq.py
> python3 split.py
# Updated
# REBASE data download and parse
### Download data from FTP
Update io yaml to have 
finput:  (fasta, input file from rebase)
final: (csv, path to final file)
temp: (tsv, path of temporary clustering data that can be later removed)
then run 
> python3 to_csv.py
