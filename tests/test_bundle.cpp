#include "../src/bundle.h"

//     phi(x) = .5 x^T A x - b^T x,
// dphi(x)/dx = Ax - b
// phi cvx if A sdp, then x sol <=> Ax = b
static int n = 4;
static MatrixXd A(n,n);
static VectorXd b(n);
static double phi(VectorXd& x, VectorXd& g) {
	g = A*x - b;
	return .5 * x.dot(A*x) - b.dot(x);
}

int main()
{
	A <<
	3, 1, 1, 0,
	1, 3, 0, 1,
	1, 0, 3, 0,
	0, 1, 0, 3;
	b << 1, 1, 1, 1;
	double diffrel = 1.;
	int nb = 0;

	bundle* phihat = new bundle(n);
	VectorXd x(n), g(n);
	double phix, phihatx;

	for(int i = 0; i < 3; ++i) {
		x = VectorXd::Random(n);
		x *= 100.0;
		phix = phi(x, g);
		phihat->addCut(x, phix, g);
	}

	while(diffrel > 1e-5 || diffrel < 0) {
		phihatx = phihat->solve(x);
		phix = phi(x, g);
		diffrel = (phix - phihatx) / abs(phix);
		std::cout << " différence relative : " << diffrel << std::endl;
		phihat->addCut(x, phix, g);
		nb++;
	}
	std::cout << "Number of iterations : " << nb << std::endl;

	std::cout << "final x = \n" << x << std::endl;
	VectorXd rx = A.inverse()*b;
	std::cout << " real x = \n" << rx << std::endl;
	std::cout << "phihat(final x) = " << phihat->eval(x) << std::endl;
	std::cout << "   phi(final x) = " << phi(x, g) << std::endl;
	std::cout << "    phi(real x) = " << phi(rx, g) << std::endl;
}
