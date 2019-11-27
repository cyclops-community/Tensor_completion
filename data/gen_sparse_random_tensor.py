import ctf
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--s',
        type=int,
        default=10,
        metavar='int',
        help='dimension (default: 10)')

    parser.add_argument(
        '--sp',
        type=float,
        default=0.1,
        metavar='float',
        help='sparsity fraction (default: .1)')
    args, _ = parser.parse_known_args()


    s = args.s
    sp = args.sp

    if ctf.comm().rank() == 0:
        print("s is",s)
        print("sp is",sp)

    

    T = ctf.tensor((s,s,s),sp=True)
    T.fill_sp_random(-1.,1.,sp)
    T.write_to_file("tensor_s" + str(s) + "_sp" + str(sp) + ".tns")

