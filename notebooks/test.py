# Test with K_4\n
G = graphs.CompleteGraph(4)
print("G order:", G.order(), "size:", G.size())
print("Regularity of G:", G.degree()[0])
r = G.degree()[0]
n = G.order()
m = G.size()
print("r =", r, "n =", n, "m =", m)
# Compute adjacency eigenvalues
A = G.adjacency_matrix()
eigs = A.eigenvalues()
print("Eigenvalues of G:", sorted(eigs, reverse=True))
# Compute line graph
L = G.line_graph()
print("L order:", L.order(), "size:", L.size())
print("Regularity of L:", L.degree()[0])
# Compute adjacency eigenvalues of L
A_L = L.adjacency_matrix()
eigs_L = A_L.eigenvalues()
print("Eigenvalues of L:", sorted(eigs_L, reverse=True))
# Predict eigenvalues using formula
pred = +[-2] * (m - n)
print("Predicted eigenvalues:", sorted(pred, reverse=True))
# Check equality (multiset)
print("Are predicted eigenvalues equal to actual eigenvalues?", sorted(pred) == sorted(eigs_L))


# Test with K_4
G = graphs.CompleteGraph(4)
print("G order:", G.order(), "size:", G.size())
print("Regularity of G:", G.degree()[0])
r = G.degree()[0]
n = G.order()
m = G.size()
print("r =", r, "n =", n, "m =", m)
# Compute adjacency eigenvalues
A = G.adjacency_matrix()
eigs = A.eigenvalues()
print("Eigenvalues of G:", sorted(eigs, reverse=True))
# Compute line graph
L = G.line_graph()
print("L order:", L.order(), "size:", L.size())
print("Regularity of L:", L.degree()[0])
# Compute adjacency eigenvalues of L
A_L = L.adjacency_matrix()
eigs_L = A_L.eigenvalues()
print("Eigenvalues of L:", sorted(eigs_L, reverse=True))
# Predict eigenvalues using formula
pred = +[-2] * (m - n)
print("Predicted eigenvalues:", sorted(pred, reverse=True))
# Check equality (multiset)
print("Are predicted eigenvalues equal to actual eigenvalues?", sorted(pred) == sorted(eigs_L))


# Random 3-regular graph of order 6
G = graphs.RandomRegular(3, 6)
print("G order:", G.order(), "size:", G.size())
print("Regularity of G:", G.degree()[0])
r = G.degree()[0]
n = G.order()
m = G.size()
print("r =", r, "n =", n, "m =", m)
# Compute adjacency eigenvalues
A = G.adjacency_matrix()
eigs = A.eigenvalues()
print("Eigenvalues of G:", sorted(eigs, reverse=True))
# Compute line graph
L = G.line_graph()
print("L order:", L.order(), "size:", L.size())
print("Regularity of L:", L.degree()[0])
# Compute adjacency eigenvalues of L
A_L = L.adjacency_matrix()
eigs_L = A_L.eigenvalues()
print("Eigenvalues of L:", sorted(eigs_L, reverse=True))
# Predict eigenvalues using formula
pred = +[-2] * (m - n)
print("Predicted eigenvalues:", sorted(pred, reverse=True))
# Check equality (multiset)
print("Are predicted eigenvalues equal to actual eigenvalues?", sorted(pred) == sorted(eigs_L))
