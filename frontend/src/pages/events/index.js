import React from 'react';
import AppLayout from '../../components/layout/AppLayout';
import EventsManager from '../../components/external/EventsManager';

const EventsPage = () => {
  return (
    <AppLayout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Event Management</h1>
          <p className="mt-2 text-sm text-gray-600">
            Manage events that may impact sales forecasts. Events are used as external regressors in the forecasting model.
          </p>
        </div>
        
        <div className="bg-white shadow rounded-lg mb-8">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">About Events</h2>
            <p className="mt-2 text-sm text-gray-600">
              Events like holidays, festivals, and local activities can significantly impact sales patterns.
              Adding events to the system helps improve forecast accuracy by incorporating these external factors.
            </p>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border border-blue-200 rounded-md p-4 bg-blue-50">
                <h3 className="font-medium text-blue-800">Holidays</h3>
                <p className="text-sm text-blue-700 mt-1">
                  National and local holidays that affect shopping behavior and business operations.
                </p>
              </div>
              <div className="border border-purple-200 rounded-md p-4 bg-purple-50">
                <h3 className="font-medium text-purple-800">Festivals</h3>
                <p className="text-sm text-purple-700 mt-1">
                  Cultural and seasonal festivals that can drive increased sales for certain product categories.
                </p>
              </div>
              <div className="border border-green-200 rounded-md p-4 bg-green-50">
                <h3 className="font-medium text-green-800">Local Events</h3>
                <p className="text-sm text-green-700 mt-1">
                  City-wide events, promotions, or activities that may result in traffic pattern changes.
                </p>
              </div>
            </div>
          </div>
        </div>
        
        <EventsManager />
      </div>
    </AppLayout>
  );
};

export default EventsPage;