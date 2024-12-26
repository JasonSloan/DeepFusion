/* in the real defination of my code, I only need to
   declare the callback function using "typedef", and 
   specify the income type and return type*/
typedef void(* CallBack)(float);

void myfunc(CallBack callback){
    // do something and get a result, for example, get a float value 100.
    // push the value to the callback function
    callback(100.);
}