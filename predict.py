from fastapi import FastAPI
from pydantic import BaseModel

from utils.utils import read_model_asset, create_feature_matrix, predict

MODEL_FILENAME = 'model.bin'
DV_FILENAME = 'DictVectorizer.bin'

model = read_model_asset(MODEL_FILENAME)
dv = read_model_asset(DV_FILENAME)


class Booking(BaseModel):
    no_of_adults: int
    no_of_weekend_nights: int
    no_of_week_nights: int
    type_of_meal_plan: str
    required_car_parking_space: int
    room_type_reserved: str
    lead_time: int
    arrival_month: int
    arrival_date: int
    market_segment_type: str
    repeated_guest: int
    no_of_previous_cancellations: int
    no_of_previous_bookings_not_canceled: int
    avg_price_per_room: float
    no_of_special_requests: int


app = FastAPI()


@app.post('/predict/')
async def predict_booking_cancellation(booking: Booking):
    booking_dict = booking.model_dump()
    X = create_feature_matrix([booking_dict], dv)
    prob = predict(X, model=model)
    return {'cancellation_prob': float(prob)}
