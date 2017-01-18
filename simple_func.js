
/**
---softmax classifier
 @author oni stephen (stevoni)
**/
//--create a constructor
var softmax_classifier=function(){

	var F=[] //store the  data set after subtracting max from it
var exp_array=[]; // holds the output of the data exponential
 this.f=function(f){//takes in data
	f.forEach(function(index){
		var init=index;
		init -= Math.max.apply(Math,f); // use apply to invoke math.max function since it can find maximum for array
		F.push(init);
	});
	return F;
};

this.exp=function(f){ //find the exponential of the input
		for(var i=0;i< f.length;i++)
		{
			var b = Math.exp(f[i]);//get exponetial of each value of an array
			exp_array.push(b);//store the result in an array

		}
		return exp_array;
};

this.sum=function(exp){ // compute the sum of the exponetial result
		var sum=0;//hold the sum value
		for(var i=0;i < exp.length; i++){// loop through the array storing the exponential result
			sum +=exp[i];
		}
		return sum;
};
//finall compute the softmax nominalization of scores
//softmax function p= exp(f) / sum(exp(f))
this.softmax_func=function(exp,sum){
		var p_array=[];
		for(var i=0;i< exp.length;i++){
				p = exp[i] / sum; //p = exp(f) / sum
				p_array.push(p);//store in an array
		}
		//returns the noriminalize result
		//remember this does not calculate the loss since we are not really aware
		//of what the output it is to be compare to input value from linear classifier
		//so to perform cross- Entropy loss 
		//you can use (-Math.log())
		return p_array;
};
};

