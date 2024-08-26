int motor1L = 8;
int motor1R = 9;
int motor1pwr = 10;

int motor2L = 13;
int motor2R = 12;
int motor2pwr = 11;
int var = 0;


void setup(){
  pinMode(motor1L,OUTPUT);
  pinMode(motor1R,OUTPUT);
  pinMode(motor1pwr,OUTPUT);

  pinMode(motor2L,OUTPUT);
  pinMode(motor2R,OUTPUT);
  pinMode(motor2pwr,OUTPUT);
}
void loop(){

  while (0<200){
    analogWrite(motor1pwr,225);
    digitalWrite(motor1L, HIGH);
    digitalWrite(motor1R, LOW);

    analogWrite(motor2pwr, 225);
    digitalWrite(motor2L, LOW);
    digitalWrite(motor2R, HIGH);
    var++;
  }
}