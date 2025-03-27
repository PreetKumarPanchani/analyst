import React, { useState, useEffect } from 'react';
import { Calendar, Plus, Edit, Trash2, X, Check, AlertTriangle } from 'lucide-react';
import { fetchApi } from '../../utils/apiConfig';

const EventsManager = () => {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [editingEvent, setEditingEvent] = useState(null);
  
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    date: '',
    type: 'event', // event, holiday, festival
    description: '',
    impact: 'medium', // low, medium, high
  });

  // Fetch events on component mount
  useEffect(() => {
    fetchEvents();
  }, []);

  const fetchEvents = async () => {
    setLoading(true);
    try {
      // Get current date and 3 months from now for date range
      const today = new Date();
      const threeMonthsLater = new Date();
      threeMonthsLater.setMonth(threeMonthsLater.getMonth() + 3);
      
      const startDate = today.toISOString().split('T')[0];
      const endDate = threeMonthsLater.toISOString().split('T')[0];
      
      const data = await fetchApi(`/api/v1/external/events?start_date=${startDate}&end_date=${endDate}`);
      setEvents(data);
    } catch (err) {
      console.error("Error fetching events:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const resetForm = () => {
    setFormData({
      name: '',
      date: '',
      type: 'event',
      description: '',
      impact: 'medium',
    });
    setEditingEvent(null);
  };

  const handleAddEvent = async (e) => {
    e.preventDefault();
    
    try {
      const payload = {
        name: formData.name,
        date: formData.date,
        type: formData.type,
        description: formData.description,
        impact: formData.impact,
      };
      
      await fetchApi('/api/v1/external/events', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      // Refresh events list
      await fetchEvents();
      
      // Reset form and hide it
      resetForm();
      setShowForm(false);
    } catch (err) {
      console.error("Error adding event:", err);
      setError(err.message);
    }
  };

  const handleUpdateEvent = async (e) => {
    e.preventDefault();
    
    try {
      const payload = {
        name: formData.name,
        date: formData.date,
        type: formData.type,
        description: formData.description,
        impact: formData.impact,
      };
      
      await fetchApi(`/api/v1/external/events/${editingEvent.name}?event_date=${editingEvent.date}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      // Refresh events list
      await fetchEvents();
      
      // Reset form and hide it
      resetForm();
      setShowForm(false);
    } catch (err) {
      console.error("Error updating event:", err);
      setError(err.message);
    }
  };

  const handleDeleteEvent = async (event) => {
    if (!confirm(`Are you sure you want to delete the event "${event.name}"?`)) {
      return;
    }
    
    try {
      await fetchApi(`/api/v1/external/events/${event.name}?event_date=${event.date}`, {
        method: 'DELETE',
      });
      
      // Refresh events list
      await fetchEvents();
    } catch (err) {
      console.error("Error deleting event:", err);
      setError(err.message);
    }
  };

  const editEvent = (event) => {
    setFormData({
      name: event.name,
      date: event.date,
      type: event.type || 'event',
      description: event.description || '',
      impact: event.impact || 'medium',
    });
    
    setEditingEvent(event);
    setShowForm(true);
  };

  // Sort events by date
  const sortedEvents = [...events].sort((a, b) => new Date(a.date) - new Date(b.date));

  // Group events by month for better visualization
  const groupedEvents = sortedEvents.reduce((groups, event) => {
    const date = new Date(event.date);
    const monthYear = `${date.toLocaleString('default', { month: 'long' })} ${date.getFullYear()}`;
    
    if (!groups[monthYear]) {
      groups[monthYear] = [];
    }
    
    groups[monthYear].push(event);
    return groups;
  }, {});

  if (loading && events.length === 0) {
    return (
      <div className="flex items-center justify-center p-12">
        <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
        <span className="ml-2">Loading events...</span>
      </div>
    );
  }
  
  return (
    <div className="bg-white rounded-lg shadow">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center">
          <Calendar className="w-5 h-5 mr-2 text-blue-500" />
          Event Management
        </h2>
        <button
          onClick={() => {
            resetForm();
            setShowForm(!showForm);
          }}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
        >
          {showForm ? (
            <>
              <X className="w-4 h-4 mr-2" />
              Cancel
            </>
          ) : (
            <>
              <Plus className="w-4 h-4 mr-2" />
              Add Event
            </>
          )}
        </button>
      </div>

      {/* Error message */}
      {error && (
        <div className="mx-6 my-4 bg-red-50 p-4 rounded-md border border-red-200">
          <div className="flex">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
          <button 
            onClick={() => setError(null)}
            className="mt-2 text-sm text-red-600 hover:text-red-800"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Event Form */}
      {showForm && (
        <div className="p-6 border-b border-gray-200">
          <form onSubmit={editingEvent ? handleUpdateEvent : handleAddEvent}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                  Event Name*
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  className="w-full p-2 border border-gray-300 rounded-md"
                  required
                />
              </div>
              
              <div>
                <label htmlFor="date" className="block text-sm font-medium text-gray-700 mb-1">
                  Date*
                </label>
                <input
                  type="date"
                  id="date"
                  name="date"
                  value={formData.date}
                  onChange={handleInputChange}
                  className="w-full p-2 border border-gray-300 rounded-md"
                  required
                />
              </div>
              
              <div>
                <label htmlFor="type" className="block text-sm font-medium text-gray-700 mb-1">
                  Type
                </label>
                <select
                  id="type"
                  name="type"
                  value={formData.type}
                  onChange={handleInputChange}
                  className="w-full p-2 border border-gray-300 rounded-md"
                >
                  <option value="event">Event</option>
                  <option value="holiday">Holiday</option>
                  <option value="festival">Festival</option>
                </select>
              </div>
              
              <div>
                <label htmlFor="impact" className="block text-sm font-medium text-gray-700 mb-1">
                  Expected Impact
                </label>
                <select
                  id="impact"
                  name="impact"
                  value={formData.impact}
                  onChange={handleInputChange}
                  className="w-full p-2 border border-gray-300 rounded-md"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
              
              <div className="md:col-span-2">
                <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  id="description"
                  name="description"
                  value={formData.description}
                  onChange={handleInputChange}
                  rows="3"
                  className="w-full p-2 border border-gray-300 rounded-md"
                ></textarea>
              </div>
            </div>
            
            <div className="mt-4 flex justify-end space-x-2">
              <button
                type="button"
                onClick={() => {
                  resetForm();
                  setShowForm(false);
                }}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
              >
                <Check className="w-4 h-4 mr-2" />
                {editingEvent ? 'Update Event' : 'Add Event'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Events List */}
      <div className="p-6">
        {sortedEvents.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Calendar className="w-12 h-12 mx-auto text-gray-400 mb-2" />
            <p>No events found for the next three months.</p>
            <button
              onClick={() => setShowForm(true)}
              className="mt-4 px-4 py-2 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200"
            >
              Add your first event
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            {Object.entries(groupedEvents).map(([monthYear, monthEvents]) => (
              <div key={monthYear}>
                <h3 className="text-md font-medium text-gray-800 mb-3">{monthYear}</h3>
                <div className="space-y-2">
                  {monthEvents.map((event, index) => {
                    const eventDate = new Date(event.date);
                    const formattedDate = eventDate.toLocaleDateString('en-US', {
                      weekday: 'short',
                      day: 'numeric',
                      month: 'short'
                    });
                    
                    return (
                      <div 
                        key={index} 
                        className="p-4 border border-gray-200 rounded-md hover:bg-gray-50 flex justify-between items-center"
                      >
                        <div className="flex-1">
                          <div className="flex items-center">
                            <span className="text-sm font-medium text-gray-500 w-24">{formattedDate}</span>
                            <span className="font-medium text-gray-900">{event.name}</span>
                            <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
                              event.type === 'holiday' ? 'bg-red-100 text-red-800' :
                              event.type === 'festival' ? 'bg-purple-100 text-purple-800' :
                              'bg-blue-100 text-blue-800'
                            }`}>
                              {event.type || 'event'}
                            </span>
                            <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
                              event.impact === 'high' ? 'bg-yellow-100 text-yellow-800' :
                              event.impact === 'low' ? 'bg-green-100 text-green-800' :
                              'bg-gray-100 text-gray-800'
                            }`}>
                              {event.impact || 'medium'} impact
                            </span>
                          </div>
                          {event.description && (
                            <p className="text-sm text-gray-600 mt-1 ml-24">{event.description}</p>
                          )}
                        </div>
                        <div className="flex space-x-2">
                          <button
                            onClick={() => editEvent(event)}
                            className="p-1 text-gray-500 hover:text-blue-600"
                            title="Edit event"
                          >
                            <Edit className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDeleteEvent(event)}
                            className="p-1 text-gray-500 hover:text-red-600"
                            title="Delete event"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default EventsManager;