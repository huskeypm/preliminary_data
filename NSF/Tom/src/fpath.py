"""Provide a class based on pathlib.Path with some enhancements.

This module only provides a subclass of a concrete path, not a pure path.

Some portions of this code are based on similar code in the pathlib module itself."""

import os
import pathlib

#Our base class is the concrete path appropriate for this system
baseclass = pathlib.WindowsPath if os.name == 'nt' else pathlib.PosixPath

class fpath(baseclass):
  """pathlib.Path subclass with enhancements
  
  The main enhancement is the distinction between files and folders,
  through the use of the binary `isFile` attribute.
  If this attribute is not specified at object creation,
  it is assumed.
  
  >>> f1 = fpath('/usr/local/bin')
  >>> f1.isFile
  False
  >>> f2 = fpath('myfile.txt.tar.gz')
  >>> f2.isFile
  True
  
  Concatenation works as for pathlib.Path,
  with interpretation of the isFile attribute.
  
  >>> f3 = f1 / f2
  >>> f3.isFile
  True
  
  The other major enhancement is a set of attributes to get
  portions of the path as strings.
  >>> assert os.name != 'nt', "These doctests were not designed for windows. Sorry."
  >>> f3
  fpath('/usr/local/bin/myfile.txt.tar.gz')
  >>> f3.fullpath #Note that this returns a string
  '/usr/local/bin/myfile.txt.tar.gz'
  >>> f3.folder
  '/usr/local/bin'
  >>> f3.filename
  'myfile.txt.tar.gz'
  
  Note how this is harder with pathlib.Path.
  >>> f1.folder
  '/usr/local/bin'
  >>> f1.parent #pathlib.Path method
  fpath('/usr/local')
  >>> f3.parent
  fpath('/usr/local/bin')
  
  The folder is also available as an fpath if desired.
  >>> f3.folder_fpath
  fpath('/usr/local/bin')
  
  File names are broken into pieces in a slightly different way than pathlib.Path,
  although the pathlib.Path methods are still available.
  >>> f3.filename_parts
  ['myfile', '.txt', '.tar', '.gz']
  >>> f3.stem #pathlib.Path method
  'myfile.txt.tar'
  >>> f3.stemname
  'myfile'
  >>> f3.suffix #pathlib.Path method
  '.gz'
  >>> f3.ext
  '.txt.tar.gz'
  """
  __slots__=('isFile')
  def __new__(cls, *args, **kwargs):
    self = cls._from_parts(args, init=False)
    if not self._flavour.is_supported: #Is this check really needed?
      raise NotImplementedError("cannot instantiate %r on your system"
                                % (cls.__name__,))
    isFile = kwargs.get('isFile',None)
    self._init(isFile)
    return self
  def _default_isFile_result(self):
    return len(self.suffix)>0
  def _init(self,isFile=None):
    baseclass._init(self)
    if isFile is None:
      #Assume anything with an extension is a file
      self.isFile = self._default_isFile_result()
    else:
      self.isFile=isFile
  def __truediv__(self,key):
    assert not self.isFile, "Cannot further append to path ending in a file."
    res=baseclass.__truediv__(self,key)
    #If the key is a file or a folder, the result will be as well.
    if hasattr(key,'isFile'):
      res.isFile=key.isFile
    else:
      #Use the default assumption
      res.isFile=res._default_isFile_result()
    return res
  def __rtruediv__(self,key):
    assert not getattr(key,'isFile',False), "Cannot append a path to a file"
    res=baseclass.__rtruediv__(self,key)
    #If self is a file or a folder, teh result will be as well.
    res.isFile=self.isFile
    return res
  @property
  def fullpath(self):
    return str(self)
  @property
  def folder_fpath(self):
    if self.isFile:
      return self.parent
    else:
      return self
  @property
  def folder(self):
    return str(self.folder_fpath)
  @property
  def foldername(self):
    return self.folder
  @property
  def filename(self):
    if not self.isFile:
      return ''
    else:
          return self.name
  @property
  def filename_parts(self):
    l=self.filename.split('.')
    for i in range(1,len(l)):
      l[i]='.'+l[i]
    return l
  @property
  def ext(self):
    #Note how this is different than self.suffix if more than one dot is used
    return ''.join(self.filename_parts[1:])
  @property
  def stemname(self):
    #Note how this is different than self.stem if more than one dot is used
    return self.filename_parts[0]
  def assure_dir(self):
    f=self.folder_fpath
    if not f.exists():
      os.makedirs(f.fullpath)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
