# CUDA Reduction implementation   
  
 M. Harris, “Optimizing parallel reduction in CUDA,” presentation packaged with CUDA Toolkit, NVIDIA Corporation (2007)    



  =======================================================================  
  == Parallel DTYPE reduction  
  =======================================================================  
  Kernel mode : 0.Basic reduction  
  Number of DTYPE : 400000000  
      size of mem : 3.20 GB  
      Total number of floating point multiplications : 1.60 Gops  
      Elaped time: 97.2505 msec  
      GFLOPS : 16.4524 gflops [Avg. of 4 time(s)]  
  Check result ...  
      [Pass] GT(800057209) == Pred(800057209)  
  =======================================================================  
  
   
   
=======================================================================  
== Parallel DTYPE reduction  
=======================================================================  
Kernel mode : 1.Blocked reduction  
Number of DTYPE : 400000000  
    size of mem : 3.20 GB  
    Total number of floating point multiplications : 1.60 Gops  
    Elaped time: 76.6518 msec  
    GFLOPS : 20.8736 gflops [Avg. of 4 time(s)]  
Check result ...  
    [Pass] GT(800057209) == Pred(800057209)  
=======================================================================  
  
   
  
=======================================================================  
== Parallel DTYPE reduction  
=======================================================================  
Kernel mode : 2.Blocked shared reduction  
Number of DTYPE : 400000000  
    size of mem : 3.20 GB  
    Total number of floating point multiplications : 1.60 Gops  
    Elaped time: 83.3996 msec  
    GFLOPS : 19.1847 gflops [Avg. of 4 time(s)]  
Check result ...  
    [Pass] GT(800057209) == Pred(800057209)  
=======================================================================  
  
  
  
=======================================================================  
== Parallel DTYPE reduction  
=======================================================================  
Kernel mode : 3.Blocked shared half reduction  
Number of DTYPE : 400000000  
    size of mem : 3.20 GB  
    Total number of floating point multiplications : 1.60 Gops  
    Elaped time: 53.0057 msec  
    GFLOPS : 30.1854 gflops [Avg. of 4 time(s)]  
Check result ...  
    [Pass] GT(800057209) == Pred(800057209)  
=======================================================================  
    
