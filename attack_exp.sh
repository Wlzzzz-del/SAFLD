# activate environment
activate dl
# SAFLD
python main.py -d EMNIST  -c 100 -k 5 -a 100 --malicious_num 20 -b 256 -t 5 --dec SAFLD
python main.py -d CIFAR10  -c 100 -k 5 -a 100 --malicious_num 20 -b 256 -t 5 --dec SAFLD

# python main.py -d EMNIST  -c 1000 -k 20 -a 100 --malicious_num 200 -b 256 -t 5 --dec SAFLD
# python main.py -d CIFAR10  -c 1000 -k 20 -a 100 --malicious_num 200 -b 256 -t 5 --dec SAFLD

# FLDNorm
python main.py -d EMNIST  -c 100 -k 5 -a 100 --malicious_num 20 -b 256 -t 5 --dec FLDNorm
python main.py -d CIFAR10  -c 100 -k 5 -a 100 --malicious_num 20 -b 256 -t 5 --dec FLDNorm

# python main.py -d EMNIST  -c 1000 -k 20 -a 100 --malicious_num 200 -b 256 -t 5 --dec FLDNorm
# python main.py -d CIFAR10  -c 1000 -k 20 -a 100 --malicious_num 200 -b 256 -t 5 --dec FLDNorm

# FLDNorm
python main.py -d EMNIST  -c 100 -k 5 -a 100 --malicious_num 20 -b 256 -t 5 --dec FLD

# python main.py -d EMNIST  -c 1000 -k 20 -a 100 --malicious_num 200 -b 256 -t 5 --dec FLD

# python main.py -d CIFAR10  -c 1000 -k 20 -a 100 --malicious_num 200 -b 256 -t 5 --dec FLD
python main.py -d CIFAR10  -c 100 -k 5 -a 100 --malicious_num 20 -b 256 -t 5 --dec FLD