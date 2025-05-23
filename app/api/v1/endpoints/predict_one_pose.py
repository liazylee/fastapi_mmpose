import io

from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response

router = APIRouter()


# pose_detector = PoseDetector()

@router.post("/predict_heatmaps",
             summary="Predict human heatmaps from image",
             description="""
    Predict human heatmaps from an uploaded image using MMPose.
    
    - **image**: Upload an image file (jpg, png) containing a person
    - Returns processed image with heatmaps
    """,
             responses={
                 200: {
                     "description": "Successful pose analysis",
                     "content": {
                         "image/jpeg": {
                             "example": "Binary image data"
                         }
                     }
                 },
                 400: {
                     "description": "Invalid input",
                     "content": {
                         "application/json": {
                             "example": {"detail": "Invalid image format"}
                         }
                     }
                 }
             }
             )
async def predict_heatmaps_endpoint(
        image: UploadFile = File(
            ...,
            description="Upload an image file (jpg, png) containing a person",
            example="person.jpg"
        )
) -> Response:
    """
    Predict human heatmaps from an uploaded image.
    
    Args:
        image: Image file containing a person
        
    Returns:
        Response: Processed image with heatmaps
    """
    # Check if the uploaded file is an image
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload an image file."
        )

    # Read the image
    image_data = await image.read()

    # Convert to PIL Image for processing
    input_image = Image.open(io.BytesIO(image_data))

    # Get visualization
    # output_image = await pose_detector.get_visualization(input_image)

    # Convert to bytes
    output_image = input_image
    output = io.BytesIO()
    output_image.save(output, format='JPEG')
    output.seek(0)

    # Return the image directly as a response
    return Response(content=output.getvalue(), media_type="image/jpeg")
