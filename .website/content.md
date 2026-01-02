helios was built by me and dhruv for the **Hackhive Hackathon (2024)**, where it went on to win the competition, and i've posted it [here](https://www.instagram.com/p/C-EehrsyX0g).

helios is an autonomous agricultural rover built for field use. it is designed to move through farmland, monitor plant health, detect diseases using machine learning, and make data about crops easier to see and understand. the rover can run on its own or be controlled remotely through a web interface, with live video, sensors, and mapping all connected into a single system.

## why this exists

modern farming needs better visibility into what is happening in the field. checking crop health manually is slow, repetitive, and easy to miss problems. i wanted to build a platform that could move on its own, look at plants closely, and turn what it sees into useful information instead of just raw footage and numbers.

so helios became a tool to observe, analyze, and support better decisions in farming.

## what helios can do

helios is built around a few core capabilities:

• autonomous navigation with optional manual override  
• live video streaming and sensor telemetry  
• machine learning–based plant disease detection  
• mapping and path visualization using google maps api  
• a web dashboard for control, viewing data, and insights  
• mechanical design files so the rover can be fabricated and assembled

the goal is not just a demo rover, but something practical and field usable.

![helios rover](https://raw.githubusercontent.com/Shaileshsaravanan/helios-rover/main/rover.png)

## how the system is structured

the project is split into clear parts to keep it maintainable:

• mechanical design files for the chassis and mounts  
• embedded firmware handling rover movement and sensors  
• core base logic for control and coordination  
• a flask web application for live control, monitoring, and ml results  

everything works together as one system, but each part is understandable on its own.

## how the rover thinks

helios captures plant images, processes them with a machine learning model, and looks for disease indicators. results are then pushed to the dashboard, alongside sensor information and camera feeds. mapping support helps visualize where the rover has been and what it has seen.

the intention is to make plant health visible, not hidden inside logs or reports.

## current direction and future work

there is still room to push this further:

• better ml models across more crop types  
• faster and more efficient edge inference  
• more advanced autonomous decision-making and behaviors  

helios is meant to grow in capability as the system improves.