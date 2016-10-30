# softmax_classifier
These code perform softmax loss on the output score
*Note*
-- These code is a replical of python code from Cs231n convolutional Neural Network by Karpathy
-- And it as been tested based on the small data(score) produce in lecture 2 on svm and sofmax by karpathy
--for more clarification and to see the python code check the link below
- [Convolutional Neural Network ](http://cs231n.github.io/linear-classify/#softmax)

//--Brief view on  softmax-classifier
softmax different from svm but does likely things in different way.softmax normalize the output scores probability. 
it takes in real value let say 'z' and output vector value in '1' and '0' and give a sum of 1.
unlike svm which does not realy care about the value of output scores since it is greater dan the margin Delta then is good to go but softmax care about the scores of the correct class to heart


*formular*
softmax--- \(f(z)\)={exp(z)}/{sum(z)}
//==Python code--'//
these is the python code used to write the Softmax_Classifier libary
import numpy as np
f=np.array([123.456.789])
f-=np.max(f)
p=np.exp(f) / np.sum(f)

--python is cool just four lines of code to get it done
--but in javascript will create most of the function ourself
##Example code
//create an object of d class
var d=new softmax_classifier();
//define your data(sores) drom linear classier
var f=d.f([-2.85,0.86,0.28]);
var f_exp=d.exp(f);
var f_sum=d.sum(f_exp);
//output the softmax 
console.log(d.softmax_func(f_exp,f_sum));

##modify
You can modify the code to your own use 
e.g you can make the code to norminalize scores without much params
you can prevent the use of:from above code
d.sum() and d.exp()
can be transform to just
d=new sotmax_classifier([123,456,789])// and get same result by editing the code

