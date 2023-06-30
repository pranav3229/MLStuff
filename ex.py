from dronekit import connect, VehicleMode, Command
from pymavlink import mavutil
import time

def run_dronekit_code():
    # Connect to the Vehicle.
    print("Connecting to vehicle...")
    connection_string = "127.0.0.1:14550"  # Replace with the appropriate connection string

    vehicle = connect(connection_string)

    # Print some vehicle attribute values
    print("Get some vehicle attribute values:")
    print(" GPS: %s" % vehicle.gps_0)
    print(" Battery: %s" % vehicle.battery)
    print(" Is Armable?: %s" % vehicle.is_armable)
    print(" System status: %s" % vehicle.system_status.state)
    print(" Mode: %s" % vehicle.mode.name)

    # Check if vehicle is armable
    while not vehicle.is_armable:
        print("Waiting for vehicle to become armable.")
        time.sleep(1)

    # Set the vehicle mode to "GUIDED"
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(20)  # Take off to target altitude

    # Wait until the vehicle reaches a safe height
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        
        if vehicle.location.global_relative_frame.alt >= 19.5:  # Trigger just below target alt.
            print("Reached target altitude")
            break

        time.sleep(1)

    # Close vehicle object
    vehicle.close()
    print("Completed")

if __name__ == '__main__':
    terminate = False

    # Run the DroneKit code
    while not terminate:
        run_dronekit_code()

        user_input = input("Enter 'terminate' to end the program: ")
        if user_input == "terminate":
            terminate = True

    print("DroneKit code completed")

