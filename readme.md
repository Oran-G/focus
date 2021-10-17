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



In code repo:
> pip install -r requirments.txt
> python3 parse.py --input (path/name to data) --output (path/name of output.parquet)