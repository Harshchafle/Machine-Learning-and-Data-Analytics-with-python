

# To sync our local DashBoard to Hugging faceSpaces, simply pass a space_id to init:
import trackio
trackio.init(project="fake-training", space_id="your-username/your-space-name", config={'configuration'})


# if we are hosting dashboarrd on Spaces, we can simply share the URL or embed it anywhere using iframe:

# <iframe src="https://huggingface.co/spaces/your-username/your-space-name?embed=true" style="width: 100%; height: 500px; border: none;" allow="accelerometer; ambient-light-sensor; camera; encrypted-media; geolocation; gyroscope; microphone; midi; payment; usb; vr; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-presentation allow-same-origin allow-scripts"></iframe>

# Note: Replace "your-username/your-space-name" with your actual Hugging Face username and space name.# You can also share the link to your Hugging Face Space directly:
# https://huggingface.co/spaces/your-username/your-space-name