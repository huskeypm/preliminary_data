"""
# Time dependent solver, single species 
Has a few features, like storing pvd,hdf files and applying source/boundary and numpy based fluxes 
"""
from dolfin import *
hdfFile = "a.h5"
import numpy as np 


eps = 1e-2 
# Define Dirichlet boundary (x = 0 or x = 1)
class daBoundary(SubDomain):
  def inside(self,x,on_boundary):
    #if on_boundary:
    #  print x      
    return on_boundary

# Nonlinear equation 
class MyEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
#        self.reset_sparsity = True
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)#, reset_sparsity=self.reset_sparsity)
#        self.reset_sparsity = False

# time [ms]
# D [um^2/ms]
def RunIt(gridRes=8,
          meshName = None, # use mesh on local directory 
          mode="boundaryFlux", # volumeSource
          outputs=True,
          dt=0.1, # [ms] 
          T = 40, # [ms]
          D = Constant(5.0) # [um^2/ms]
          ):
  ## params 
  import numpy as np
  j= 0.1 # [uM/ms] 
  t = 0.0
  initConc = 1. # [uM] 
  
  ## validation
  refConcChange = T*j # [uM]
  #print refConcChange
  

  # doesn't seem to work for parallel runs?
  marker = 111
  if meshName!=None:
    mesh = Mesh(meshName)
    boundaries = MeshFunction("size_t", mesh, 1) 
    outer = daBoundary()
    outer.mark(boundaries,marker)


  else: 
    mesh = UnitSquareMesh(10,10) 
    #mesh = UnitSquareMesh(2,2) 
    mesh.coordinates()[:] = mesh.coordinates()[:]*10
    #mesh = UnitSquareMesh(100,100) 
    #mesh.coordinates()[:] = mesh.coordinates()[:]*100
    boundaries = MeshFunction("size_t", mesh, 1) 
    outer = daBoundary()
    outer.mark(boundaries,marker)

    # print for comparison 
    File("mesh.xml") << mesh
    File("bound.xml") << boundaries

  dx = Measure("dx",domain=mesh)
  ds = Measure("ds",subdomain_data=boundaries,domain=mesh)
  vol = assemble(Constant(1.)*dx)
  area = assemble(Constant(1.)*ds(marker))
  print "Vol ", vol, " Area ", area
  
  ## Define boundary condition 
  #u0 = Constant(1.0)
  #bcs=[]
  #bcs.append(DirichletBC(V, u0, subdomains,marker))
  V = FunctionSpace(mesh,"CG",1)
  
  # Define trial and test functions
  du    = TrialFunction(V)
  q  = TestFunction(V)
  # Define functions
  u   = Function(V)  # current solution
  u0  = Function(V)  # solution from previous converged step

  ## Attributes
  vol = assemble(Constant(1.)*dx(domain=mesh))
  area= assemble(Constant(1.)*ds(domain=mesh))
  boundaryArea= assemble(Constant(1.)*ds(marker,domain=mesh))
  #print "A,bA, V ", area, boundaryArea,vol

  ## Init Cond 
  nonunif= False
  if nonunif:
    ic = Expression("exp(-(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])/0.5)")
    u.interpolate(ic)
    u0.interpolate(ic)
  else:
    ic = Constant(initConc)
    u.interpolate(ic) 
    u0.interpolate(ic)


  # store data 
  if outputs:
    File("test.pvd") << u
  
  ## RHS weak form 
  # Diffusion
  RHS = -inner(D*grad(u), grad(q))*dx
  # fluxes
  if mode=="volumeSource":
    RHS += j*q*dx 
  elif mode=="boundaryFlux":
    RHS += j*(vol/boundaryArea)*q*ds(marker)
  elif mode=="numpy": # Adding flux changes directly to u vector
    1 

  ## LHS 
  # from u1 = u0 + du/dt*del_t ==> (u1-u0) - du/dt * dt = 0 
  L = (u-u0)*q*dx - dt * RHS

  
  # Compute directional derivative about u in the direction of du (Jacobian)
  a = derivative(L, u, du)
  
  
  problem = MyEquation(a,L)
  solver = NewtonSolver()#"gmres")         

  if outputs:
    ## Paraview-friendly output 
    file = File("output.pvd", "compressed")
    file << (u,t) 

    ## more robust output via hdf5 
    hdf=HDF5File(mesh.mpi_comm(), hdfFile, "w")
    hdf.write(mesh, "mesh")
  
  ## timing (probably not best practice here....)
  import datetime
  tstart = datetime.datetime.now()  

  ## Main loops 
  i=0
  while (t < T):
      t += dt
      u0.vector()[:] = u.vector()
      solver.solve(problem, u.vector())

      if mode=="numpy":
        delu = u.vector()[:] * 0  # THIS IS A HACK
          
        def uDependentFunction(uvec,j,dt):
          # this is a useless step, but just showing that we can 
          # manipulate uvec directly
          ures = uvec + j*dt # [uM] + [uM/ms]*[ms]
          return ures

        ures = uDependentFunction(u.vector()[:],j,dt)
        u.vector()[:] = ures  

      r = u.vector()
      #print "Assemble %d/%f" % (i,assemble(u*dx)) # mesh=mesh)
      conci = assemble(u*dx)/vol
      if MPI.rank(mpi_comm_world())==0:
        print "Conc[%d]: %f, t=%f" % (i,t,conci)
      
  
      if outputs:
        file << (u,t) 
        hdf.write(u, "u",i)
        #if MPI.rank(mpi_comm_world())==0:
        x = Function(V)
        x.vector()[:] = t  # not sure why I have to do it this way 
        hdf.write(x,"t",i)    


      # update 
      i+=1
  ## Loop end
  tstop = datetime.datetime.now()  
  if MPI.rank(mpi_comm_world())==0:
    print "Elapsed seconds ", (tstop-tstart).total_seconds()
  

  import numpy as np
  #hdf.write(np.array([1.,2.]),"nEntries") 
  if outputs: 
    hdf.close()

  concChange = conci - initConc 

def ReadIt(hdfFile):
  hdf = HDF5File(mpi_comm_world(),hdfFile,'r')
  mesh = Mesh()
  hdf.read(mesh,"mesh",False)
  print "Reading ", hdfFile
  #f.read(z,"nEntries") 
  V = FunctionSpace(mesh,"CG",1)
  u = Function(V)


  attr = hdf.attributes("u")
  nsteps = attr['count']
  assembled = np.zeros( nsteps ) 
  for i in range(nsteps):
    name = "u/vector_%d"%i
    hdf.read(u,name)
    print "Assemble %d/%f" % (i,assemble(u*dx)) ##,mesh=mesh)
    #vol = assemble(Constant(1.)*dx)
    #conc = assemble(u*dx) / vol 
    # store value 
    assembled[i] = assemble(u*dx)

    # ts 
    dataset = "t/vector_%d"%i
    attr = hdf.attributes(dataset)
    hdf.read(u, dataset)
    t = u.vector()[0]
    print t


  hdf.close()
  return assembled 
  
  
#!/usr/bin/env python
import sys
##################################
#

# Revisions
#       10.08.10 inception
#
##################################


#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -runMPI/-readSingle" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line 
#
if __name__ == "__main__":
  import sys
  msg = helpmsg()

  meshName = None
  outputs=True  
  T = 10 # [ms]  
  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-runMPI"):
      RunIt(outputs=outputs,T=T,mode="boundaryFlux",meshName=meshName)
    if(arg=="-mesh"):
      meshName=sys.argv[i+1]
    if(arg=="-readSingle"):
      ReadIt(hdfFile)
  


