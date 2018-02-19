#!/usr/bin/env python3
import argparse
import sys
import io

def print_vcf(theta, numsites = 100000, depth = 100):
    headerstr = """##fileformat=VCFv4.2
##contig=<ID=1,length={}
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=AD,Number=2,Type=Integer,Description="Allele Depths, ref and alt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1"""
    headerstr = headerstr.format(numsites)

    print(headerstr)
    num_refalts = round(theta * numsites)
    num_refrefs = numsites - num_refalts
    for i in range(1,num_refalts+1):
        print(lineformat(i,"0/1"))
    for i in range(num_refalts+1,num_refalts+1+num_refrefs):
        print(lineformat(i,"0/0"))

def lineformat(pos, gt):
    pos = str(pos)
    dp = 100
    ad = ",".join([str(int(dp/2)),str(int(dp/2))]) if gt == "0/1" else str(dp) + ",0"
    dp = str(dp)
    return "\t".join(["1",pos,pos,"A","T","40","PASS",".","GT:DP:AD",":".join([gt,dp,ad])])

def parse_args():
    parser = argparse.ArgumentParser(description = "Simulate a VCF with a given theta value")
    parser.add_argument("-t","--theta", required = True, type=float, help = "Theta value to simulate")
    return parser

def main():
    args = parse_args()
    args = args.parse_args()
    print_vcf(args.theta)

if __name__ == '__main__':
    main()

