"""Call dolfin-convert to convert .msh files to .xml, based on derivative of MeshParameters"""

#Standard library
from __future__ import print_function, division #Python 2 compatibility
import os
import os.path as osp
from subprocess import call
import sys

#Site packages

#Local
import folderstructure as FS
import common
import buildgeom

#Path to this code file (for dependency list)
thisfile=sys.modules[__name__].__file__

#Template for dolfin-convert command
#arguments: mshfile, xmlfile
cmd_tmpl="dolfin-convert %s %s"

class DolfinConvertRunner(common.ParameterSet):
  """Subclass of common.ParameterSet for converting gmsh .msh file to FEniCS XML format

  Attributes:
  
    To be read in:
    
      - meshname, geomdefname, tmplvalues as for buildgeom.MeshParameters
    
    To be generated by methods:
    
      - mshfile = name of input .msh file (not full path)
      - xmlfile = name of XML file storing mesh output (not full path)
    
    The names of the MeshFunction output files are not stored in any attribute here.
  """
  __slots__=('meshname','geomdefname','tmplvalues','mshfile','xmlfile','_folders')
  _required_attrs=['meshname','geomdefname','tmplvalues']
  _config_attrs=_required_attrs
  #don't need sourcefile as input file due to config
  _inputfile_attrs=['mshfile']
  _more_inputfiles=[thisfile,common.__file__]
  _outputfile_attrs=['xmlfile']
  _taskname_src_attr='meshname'

  def __init__(self,**kwd):
    #Initialization from base class
    super(DolfinConvertRunner, self).__init__(**kwd)
    #Get folders
    self._folders={'mshfile':osp.join(FS.mshfolder,self.basename),
                   'xmlfile':osp.join(FS.xmlfolder,self.basename)}
    #Get name of input and output files
    self.mshfile=self.meshname+'.msh'
    self.xmlfile=self.meshname+'.xml'

  def run(self):
    print(self.xmlfile)
    #Create directories if necessary
    for oattr in self._outputfile_attrs:
      if not osp.isdir(self._folders[oattr]):
        os.makedirs(self._folders[oattr])
    #Run the shell command
    cmd_str=cmd_tmpl%(self.full_path('mshfile'),self.full_path('xmlfile'))
    call(cmd_str,shell=True)

#Support command-line arguments
if __name__ == '__main__':
  program_description='Create dolfin mesh .xml file(s) from .msh file(s) by running dolfin-convert from a yaml input file'
  input_file_description="""Path to parameter definition file for the mesh
    This is a potentially multi-doc yaml file, where each document specifies one mesh to generate."""
  
  common.run_cmd_line(program_description,input_file_description,DolfinConvertRunner)
