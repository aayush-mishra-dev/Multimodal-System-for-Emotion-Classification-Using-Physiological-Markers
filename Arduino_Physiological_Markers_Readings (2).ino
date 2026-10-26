

#define SAMPLE_RATE 184 
#define BAUD_RATE 115200
// #define INPUT_PIN1 32
// #define INPUT_PIN2 35

#define INPUT_PIN1 A1
#define INPUT_PIN2 A0
#define INPUT_PIN3 A4

void setup() {
	// Serial connection begin
	Serial.begin(BAUD_RATE);
}

void loop() {
	// Calculate elapsed time
	static unsigned long past = 0;
	unsigned long present = micros();
	unsigned long interval = present - past;
	past = present;

	// Run timer
	static long timer = 0;
	timer -= interval;

	// Sample
	if(timer < 0){
		timer += 1000000 / SAMPLE_RATE;
		float sensor_value1 = analogRead(INPUT_PIN1);
    float sensor_value2 = analogRead(INPUT_PIN2);
		float sensor_value3 = analogRead(INPUT_PIN3);

  //USE THIS FOR NON FILTTERED SIGNAL
		int signal1 = (sensor_value1);
    int signal2 = (sensor_value2);
    int signal3 = (sensor_value3);
    //USE THIS FOR FILTTERD SIGNAL
		// int signal1 = ECGFilter(sensor_value1);
    // int signal2 = ECGFilter(sensor_value2);
    
		Serial.print(signal1);
    Serial.print(",");
    Serial.print(signal2);
		Serial.print(",");
		Serial.println(signal3);
	}
}
