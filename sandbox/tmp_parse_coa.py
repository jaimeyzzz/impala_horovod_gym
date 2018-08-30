""" script that reads in the CSV exported from c.oa.com and outputs the CSV
needed by run_experiment_mm_raw.py

Example:
run_experiment_mm_raw.py a.csv b.csv
 """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import sys

def write_cluster_csv(worker_list, cluster_csv_path):
    with open(cluster_csv_path, 'wb') as fp:
        spamwriter = csv.writer(fp, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(['job', 'ip', 'tf_port', 'cuda_visible_devices', 'ssh_port', 'ssh_username', 'ssh_password'])

        center_ip = '100.102.32.34'
        spamwriter.writerow(['ps', center_ip, '8000', '', '', '', ''])
        spamwriter.writerow(['learner', center_ip, '8001', '', '', '', ''])
        spamwriter.writerow(['learner', center_ip, '8002', '', '', '', ''])

        for worker in worker_list:
            ip, port, user, passwd = worker
            spamwriter.writerow(['actor', ip, '8000', '', port, user, passwd])


def read_coa_csv(coa_csv_path):
    with open(coa_csv_path, 'rb') as fp:
        spamreader = csv.reader(fp, delimiter=',', quotechar='|')
        worker_list = [(row[1], row[2], 'root', 'Server@Teg56243') for row in spamreader][1:]
    return worker_list

def main():

    coa_csv_path = sys.argv[1]
    cluster_csv_path = sys.argv[2]

    worker_list = read_coa_csv(coa_csv_path)

    write_cluster_csv(worker_list, cluster_csv_path)

if __name__ == '__main__':
    main()
