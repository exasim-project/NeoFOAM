# import networkx as nx
# from foamadapter.framework.dag import build_step_dag, build_global_dag
# from foamadapter.framework.decorators import model, step
# import networkx as nx
# import os
# import matplotlib.pyplot as plt

# PLOT_DAG = False

# def test_interdomain_dependencies():

#     @model("DomainA")
#     class DomainA:
#         @step(order=1)
#         def a1(self, ctrl): pass
#         @step(order=2, depends_on=["domainb.b1"])
#         def a2(self, ctrl): pass

#     @model("DomainB")
#     class DomainB:
#         @step(order=1)
#         def b1(self, ctrl): pass
#         @step(order=2, depends_on=["domaina.a1"])
#         def b2(self, ctrl): pass



#     domains = {"domaina": DomainA(), "domainb": DomainB()}
#     dag = build_global_dag(domains)
#     # Interdomain edges
#     assert ("domainb.b1", "domaina.a2") in dag.edges()
#     assert ("domaina.a1", "domainb.b2") in dag.edges()
#     # Topological order: b1 before a2, a1 before b2
#     order = list(nx.topological_sort(dag))
#     assert order.index("domainb.b1") < order.index("domaina.a2")
#     assert order.index("domaina.a1") < order.index("domainb.b2")


#     # Set this flag to True to plot DAGs during tests
#     # PLOT_DAG = bool(os.environ.get("PLOT_DAG", False))

#     if PLOT_DAG:
#         plt.figure()
#         pos = nx.spring_layout(dag)
#         nx.draw(dag, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
#         plt.title("Interdomain DAG")
#         plt.show()


# def test_global_dag_cross_domain():


#     @model("Fluid")
#     class Fluid:
#         @step(order=1)
#         def momentum(self, ctrl): pass
#         @step(order=2, depends_on=["solid.conduction"])
#         def energy_exchange(self, ctrl): pass

#     @model("Solid")
#     class Solid:
#         @step(order=1)
#         def conduction(self, ctrl): pass
#         @step(order=2, depends_on=["fluid.momentum"])
#         def heat_transfer(self, ctrl): pass

#     # Build global DAG
#     from foamadapter.framework.dag import build_global_dag

#     domains = {"fluid": Fluid(), "solid": Solid()}
#     dag = build_global_dag(domains)
#     # Check edges
#     assert set(dag.edges()) == {
#         ("solid.conduction", "fluid.energy_exchange"),
#         ("fluid.momentum", "solid.heat_transfer"),
#     }
#     # Check topological order
#     order = list(nx.topological_sort(dag))
#     # conduction must come before energy_exchange, momentum before heat_transfer
#     assert order.index("solid.conduction") < order.index("fluid.energy_exchange")
#     assert order.index("fluid.momentum") < order.index("solid.heat_transfer")

#     if PLOT_DAG:
#         plt.figure()
#         pos = nx.spring_layout(dag)
#         nx.draw(dag, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=2000, font_size=10)
#         plt.title("Global Cross-Domain DAG")
#         plt.show()



# def test_step_dag_linear():
#     @model("LinearModel")
#     class LinearModel:
#         @step(order=1)
#         def a(self, ctrl): pass
#         @step(order=2)
#         def b(self, ctrl): pass
#         @step(order=3)
#         def c(self, ctrl): pass

#     dag = build_step_dag(LinearModel())
#     assert list(nx.topological_sort(dag)) == ["a", "b", "c"]

#     if PLOT_DAG:
#         plt.figure()
#         pos = nx.spring_layout(dag)
#         nx.draw(dag, pos, with_labels=True, node_color='lightyellow', edge_color='gray', node_size=2000, font_size=10)
#         plt.title("Linear Step DAG")
#         plt.show()

# def test_step_dag_with_dependencies():
#     @model("DepModel")
#     class DepModel:
#         @step(order=1)
#         def a(self, ctrl): pass
#         @step(order=2, depends_on=["a"])
#         def b(self, ctrl): pass
#         @step(order=3, depends_on=["b"])
#         def c(self, ctrl): pass
#         @step(order=4, depends_on=["a"])
#         def d(self, ctrl): pass

#     dag = build_step_dag(DepModel())
#     # c depends on b, b depends on a, d depends on a
#     assert set(dag.edges()) == {("a", "b"), ("b", "c"), ("a", "d")}
#     assert list(nx.topological_sort(dag)) == ["a", "b", "d", "c"] or ["a", "d", "b", "c"]

#     if PLOT_DAG:
#         plt.figure()
#         pos = nx.spring_layout(dag)
#         nx.draw(dag, pos, with_labels=True, node_color='lightcoral', edge_color='gray', node_size=2000, font_size=10)
#         plt.title("Step DAG with Dependencies")
#         plt.show()