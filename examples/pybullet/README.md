## teleOP
for teleOP add your trajectory file in utils. and name it as traj_target in numpy mpy extension. \\

the data will be in the format of : \\
[x,y,z,r,p,y,q1,q2,q3,q4] \\
where x,y,z are the position of the end effectoe, \\
r,p,y are the euler angles, \\
q1,q2,q3,q4 are the quaternions with q1 being the real part of the quaternion.