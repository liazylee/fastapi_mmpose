# from io import BytesIO
#
# import cv2
# import numpy as np
# from fastapi import APIRouter, UploadFile, File
# from fastapi.responses import StreamingResponse
#
# from app.pose_service.core import pose_service
#
# router = APIRouter()
#
#
# @router.post("/predict_pose",
#              summary="Predict human pose from image",
#              description="""
#     Predict human pose from an uploaded image using MMPose.
#
#     - **image**: Upload an image file (jpg, png) containing a person
#     - Returns processed image with pose visualization
#     """,
#              responses={
#                  200: {
#                      "description": "Successful pose analysis",
#                      "content": {
#                          "image/jpeg": {
#                              "example": "Binary image data"
#                          }
#                      }
#                  },
#                  400: {
#                      "description": "Invalid input",
#                      "content": {
#                          "application/json": {
#                              "example": {"detail": "Invalid image format"}
#                          }
#                      }
#                  }
#              }
#              )
# async def predict_pose_endpoint(
#         image: UploadFile = File(
#             ...,
#             description="Upload an image file (jpg, png) containing a person",
#             example="person.jpg"
#         )
# ) -> StreamingResponse:
#     """
#     Predict human pose from an uploaded image.
#
#     Args:
#         image: Image file containing a person
#
#     Returns:
#         StreamingResponse: Processed image with pose visualization
#     """
#     # Read image from upload
#     data = await image.read()
#     np_arr = np.frombuffer(data, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#
#     # Process image through pose service
#     vis_img, pose_results = pose_service.process_image(img)
#     # Encode to JPEG and return
#     _, jpeg = cv2.imencode('.jpg', vis_img)
#     buf = BytesIO(jpeg.tobytes())
#     return StreamingResponse(buf, media_type='image/jpeg')
