from dataclasses import dataclass, field
from typing import List
import math

@dataclass
class samp1d():
    file_config:str # .yaml config file.
    #pop_sim: List[float] = field(default_factory=list) # list of poppulation values.     
    nsim:int = 0
    #shape: float = 0.0
    name:str = None
    chunk_size: int = 0
    percent_feed: int = 30
    samp_chuck: List[float] = field(default_factory=list) # list of samples values.     
    
    def split2chunk(self,pop_sim:list,chunk_size:int = None):
        if chunk_size is None:
            chunk_size  = 50
        self.chunk_size = chunk_size
        pop_n = len(pop_sim)
        samp_num = math.ceil((self.percent_feed/100)*pop_n)
        num_ch = math.floor(samp_num/self.chunk_size)
        list_chunk = [pop_sim[i:i + self.chunk_size] for i in range(0,(self.chunk_size*(num_ch-1)),self.chunk_size)]
        list_chunk.append(pop_sim[(self.chunk_size*(num_ch-1)):(self.chunk_size*(num_ch-1)) + self.chunk_size])
        self.samp_chuck = list_chunk