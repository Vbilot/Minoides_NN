"""autogenerated by genmsg_py from Log.msg. Do not edit."""
import roslib.message
import struct

import roslib.msg

class Log(roslib.message.Message):
  _md5sum = "acffd30cd6b6de30f120938c17c593fb"
  _type = "roslib/Log"
  _has_header = True #flag to mark the presence of a Header object
  _full_text = """##
## Severity level constants
##
byte DEBUG=1 #debug level
byte INFO=2  #general level
byte WARN=4  #warning level
byte ERROR=8 #error level
byte FATAL=16 #fatal/critical level
##
## Fields
##
Header header
byte level
string name # name of the node
string msg # message 
string file # file the message came from
string function # function the message came from
uint32 line # line the message came from
string[] topics # topic names that the node publishes

================================================================================
MSG: roslib/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.secs: seconds (stamp_secs) since epoch
# * stamp.nsecs: nanoseconds since stamp_secs
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

"""
  # Pseudo-constants
  DEBUG = 1
  INFO = 2
  WARN = 4
  ERROR = 8
  FATAL = 16

  __slots__ = ['header','level','name','msg','file','function','line','topics']
  _slot_types = ['Header','byte','string','string','string','string','uint32','string[]']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.
    
    The available fields are:
       header,level,name,msg,file,function,line,topics
    
    @param args: complete set of field values, in .msg order
    @param kwds: use keyword arguments corresponding to message field names
    to set specific fields. 
    """
    if args or kwds:
      super(Log, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.header is None:
        self.header = roslib.msg._Header.Header()
      if self.level is None:
        self.level = 0
      if self.name is None:
        self.name = ''
      if self.msg is None:
        self.msg = ''
      if self.file is None:
        self.file = ''
      if self.function is None:
        self.function = ''
      if self.line is None:
        self.line = 0
      if self.topics is None:
        self.topics = []
    else:
      self.header = roslib.msg._Header.Header()
      self.level = 0
      self.name = ''
      self.msg = ''
      self.file = ''
      self.function = ''
      self.line = 0
      self.topics = []

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    @param buff: buffer
    @type  buff: StringIO
    """
    try:
      _x = self
      buff.write(_struct_3I.pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      buff.write(_struct_b.pack(self.level))
      _x = self.name
      length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.msg
      length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.file
      length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.function
      length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      buff.write(_struct_I.pack(self.line))
      length = len(self.topics)
      buff.write(_struct_I.pack(length))
      for val1 in self.topics:
        length = len(val1)
        buff.write(struct.pack('<I%ss'%length, length, val1))
    except struct.error as se: self._check_types(se)
    except TypeError as te: self._check_types(te)

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    @param str: byte array of serialized message
    @type  str: str
    """
    try:
      if self.header is None:
        self.header = roslib.msg._Header.Header()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _struct_3I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.header.frame_id = str[start:end]
      start = end
      end += 1
      (self.level,) = _struct_b.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.name = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.msg = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.file = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.function = str[start:end]
      start = end
      end += 4
      (self.line,) = _struct_I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.topics = []
      for i in xrange(0, length):
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        val1 = str[start:end]
        self.topics.append(val1)
      return self
    except struct.error as e:
      raise roslib.message.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    @param buff: buffer
    @type  buff: StringIO
    @param numpy: numpy python module
    @type  numpy module
    """
    try:
      _x = self
      buff.write(_struct_3I.pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      buff.write(_struct_b.pack(self.level))
      _x = self.name
      length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.msg
      length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.file
      length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.function
      length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      buff.write(_struct_I.pack(self.line))
      length = len(self.topics)
      buff.write(_struct_I.pack(length))
      for val1 in self.topics:
        length = len(val1)
        buff.write(struct.pack('<I%ss'%length, length, val1))
    except struct.error as se: self._check_types(se)
    except TypeError as te: self._check_types(te)

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    @param str: byte array of serialized message
    @type  str: str
    @param numpy: numpy python module
    @type  numpy: module
    """
    try:
      if self.header is None:
        self.header = roslib.msg._Header.Header()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _struct_3I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.header.frame_id = str[start:end]
      start = end
      end += 1
      (self.level,) = _struct_b.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.name = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.msg = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.file = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.function = str[start:end]
      start = end
      end += 4
      (self.line,) = _struct_I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.topics = []
      for i in xrange(0, length):
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        val1 = str[start:end]
        self.topics.append(val1)
      return self
    except struct.error as e:
      raise roslib.message.DeserializationError(e) #most likely buffer underfill

_struct_I = roslib.message.struct_I
_struct_3I = struct.Struct("<3I")
_struct_b = struct.Struct("<b")
