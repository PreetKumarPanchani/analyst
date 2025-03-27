// Import necessary libraries
import { utils, write } from 'xlsx';
import Papa from 'papaparse';
import fs from 'fs';

// Helper function to generate UUID
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// Helper function to format date and time
function formatDate(date) {
  return date.toISOString().split('T')[0];
}

function formatTime(date) {
  return date.toTimeString().slice(0, 8);
}

function formatDateTime(date) {
  return `${formatDate(date)} ${formatTime(date)}`;
}

// Function to generate a random time after 15:40:00
function generateRandomTimeAfter1540(date) {
  const hours = Math.floor(Math.random() * (23 - 15)) + 15;
  const minutes = hours === 15 ? 
    Math.floor(Math.random() * (60 - 40)) + 40 : 
    Math.floor(Math.random() * 60);
  const seconds = Math.floor(Math.random() * 60);
  
  date.setHours(hours, minutes, seconds);
  return date;
}

// Sample data from existing sheets
const products = [
  { id: '3ef3c588-272c-4594-a6ca-c58f85b1dc85', name: 'Croissant', category: 'Pastry', price: 2.85 },
  { id: '95cba43d-c401-4c24-836f-f8d64c57885a', name: 'Chocolate orange Pain Suisse', category: 'Pastry', price: 3.8 },
  { id: '02669149-9c33-4038-860d-c046788eab44', name: 'Pain au Chocolat', category: 'Pastry', price: 3.8 },
  { id: 'e3bff61c-ba89-4ade-88b6-65ec6166d66f', name: 'Doughnut - Custard', category: 'Pastry', price: 3.7 },
  { id: 'd60c0fb9-c0f5-4957-b598-4e6bdb8453ab', name: 'Maple Pecan Swirl', category: 'Pastry', price: 3.6 },
  { id: 'bc3a3ac4-cd6f-45df-9cd4-8003e03bb336', name: 'Espresso', category: 'Hot Drinks', price: 2.8 },
  { id: '3c81c71d-5f7d-4210-970e-357eaa099ca5', name: 'White Peak - Large', category: 'Bread', price: 4.6 },
  { id: 'c6cb24c1-6424-4349-9eb3-990a351fc228', name: 'Latte', category: 'Hot Drinks', price: 3.6 },
  { id: '978a09c4-aa56-48a8-92ab-6ae679f0c4fa', name: 'Festive Swirls', category: 'Pastry', price: 3.5 },
  { id: 'ca498703-acd6-4b99-9629-d0c5b3eba960', name: 'Bear Claw', category: 'Pastry', price: 3.5 },
  { id: '523ff915-062d-4d87-be94-668625cbe68c', name: 'Mocha', category: 'Hot Drinks', price: 4.1 },
  { id: '473a9733-36fa-4bab-b8e6-5c2e58667f15', name: 'Pork and Fennel Sausage Roll', category: 'Cafe Counter', price: 4.5 },
  { id: '5c41053f-09c1-48d2-a67c-897dede4a092', name: 'Almond Croissant', category: 'Pastry', price: 3.8 },
  { id: 'fb97be39-087f-4133-9b5e-68f68f2b0aed', name: 'Light Rye - Large', category: 'Bread', price: 4.6 },
  { id: '78cc9710-ef82-4569-86fb-e50d1949fa60', name: 'Americano', category: 'Hot Drinks', price: 3.2 }
];

const staff = [
  'Aaron Mills', 'Harry', 'Sasha', 'Eleanor Paul', 'Georgia Hatton', 'Rachel Glover', 'Acacia Williams-Hewitt'
];

const registers = ['Train Station', 'Bakery Counter'];
const cardBrands = ['VISA', 'MASTERCARD'];
const paymentMethods = ['SUMUP'];
const entryModes = ['contactless', 'chip'];

// Generate sales for 3 days
const startDate = new Date('2025-02-22');
const numSales = 30; // Total number of sales across 3 days

// Arrays to store data for each sheet
const sheet1Data = []; // Main transactions
const sheet2Data = []; // Line items
const sheet3Data = []; // Payment details
const sheet4Data = []; // Refunds
const sheet5Data = []; // Additional product details

// Generate sale data
for (let i = 0; i < numSales; i++) {
  // Determine which day (0, 1, or 2 days after start date)
  const dayOffset = Math.floor(i / 10);
  const saleDate = new Date(startDate);
  saleDate.setDate(startDate.getDate() + dayOffset);
  
  // Generate time after 15:40:00
  generateRandomTimeAfter1540(saleDate);
  
  const saleId = generateUUID();
  const register = registers[Math.floor(Math.random() * registers.length)];
  const staffMember = staff[Math.floor(Math.random() * staff.length)];
  
  // Determine number of items in this sale (1-4)
  const numItems = Math.floor(Math.random() * 4) + 1;
  
  // Calculate total for this sale
  let saleTotal = 0;
  const saleProducts = [];
  
  for (let j = 0; j < numItems; j++) {
    const product = products[Math.floor(Math.random() * products.length)];
    saleProducts.push(product);
    saleTotal += product.price;
  }
  
  // Round to 2 decimal places
  saleTotal = parseFloat(saleTotal.toFixed(2));
  
  // Create Sheet 1 record (main transaction)
  sheet1Data.push({
    'Sale ID': saleId,
    'Sale Date': formatDate(saleDate),
    'Sale Time': formatTime(saleDate),
    'Sale Type': 'INSTORE',
    'Order Status': 'COMPLETED',
    'Outlet': 'BAKERY',
    'Outlet Tag': 'Bakery',
    'Customer Name': '',
    'Customer Email': '',
    'Receipt No': (1.73581E+12 + i).toString(),
    'Order No': (i + 1).toString(),
    'Goodeats Order Reference': '',
    'Register': register,
    'Staff': staffMember,
    'Coupon Amount': '',
    'Promotion Amount': '',
    'Line Discount': '',
    'Sale Discount': '',
    'Quantity': numItems,
    'Net': saleTotal,
    'VAT': 0,
    'Service Charge': 0,
    'Delivery Charge': 0,
    'Total': saleTotal,
    'Tips': '',
    'Goodeats Transaction fee': '',
    'Goodeats Refund': '',
    'Goodeats Refunded AT': '',
    'Covers': '',
    'Order Notes': '',
    'Split By': '',
    'Split For': '',
    'Eat-in/Takeaway': 'Takeaway',
    'Table No': '',
    'Table Opened At': '',
    'Table Opened By': '',
    'Table Opened At Register': '',
    'Table Billed At': '',
    'Table Billed By': '',
    'Booking Provider': '',
    'Booking Reference': '',
    'Fulfilment Type': '',
    'Collection/Dropoff point': '',
    'Goodeats Accepted At': '',
    'Goodeats Finished At': '',
    'Customer Group': ''
  });
  
  // Create Sheet 3 record (payment details)
  sheet3Data.push({
    'Sale ID': saleId,
    'Sale Date': formatDate(saleDate),
    'Sale Time': formatTime(saleDate),
    'Order Status': 'COMPLETED',
    'Register': register,
    'Staff': staffMember,
    'Payment Date': formatDateTime(saleDate),
    'Payment Method': paymentMethods[0],
    'Payment Amount': saleTotal,
    'Tips': '',
    'Reference Number': 'T' + Math.random().toString(36).substring(2, 11).toUpperCase(),
    'Card Brand': cardBrands[Math.floor(Math.random() * cardBrands.length)],
    'Card Entry Mode': entryModes[Math.floor(Math.random() * entryModes.length)],
    'Payment Description': 'SumUp'
  });
  
  // Create Sheet 2 records (line items, one per product)
  saleProducts.forEach(product => {
    sheet2Data.push({
      'Sale ID': saleId,
      'Sale Date': formatDate(saleDate),
      'Sale Time': formatTime(saleDate),
      'Order Status': 'COMPLETED',
      'Register': register,
      'Staff': staffMember,
      'Product ID': product.id,
      'Product Name': product.name,
      'Category': product.category,
      'Item Type': 'Product',
      'Quantity': 1,
      'Unit Price': product.price,
      'Line Discount': 0,
      'Sale Discount': 0,
      'VAT Description': 'NO VAT',
      'VAT Rate': 0,
      'Net': product.price,
      'VAT': 0,
      'Total': product.price,
      'Eat-in/Takeaway': 'Takeaway',
      'Item Notes': '',
      'Promotion': '',
      'Account Code': ''
    });
    
    // Also add to Sheet 5 (additional product details)
    sheet5Data.push({
      'Sale ID': saleId,
      'Sale Date': formatDate(saleDate),
      'Sale Time': formatTime(saleDate),
      'Order Status': 'COMPLETED',
      'Register': register,
      'Staff': staffMember,
      'Product ID': product.id,
      'Product Name': product.name,
      'Category': product.category,
      'Item Type': 'Product',
      'Quantity': 1,
      'Unit Price': product.price,
      'VAT Description': 'NO VAT',
      'VAT Rate': 0,
      'Total': product.price,
      'Eat-in/Takeaway': 'Takeaway',
      'Item Notes': '',
      'Account Code': ''
    });
  });
  
  // Randomly decide if this sale should have a refund (about 10% chance)
  if (Math.random() < 0.1) {
    const refundDate = new Date(saleDate);
    refundDate.setHours(refundDate.getHours() + Math.floor(Math.random() * 24) + 1); // Refund 1-24 hours later
    
    const refundAmount = parseFloat((saleTotal * (Math.random() * 0.5 + 0.5)).toFixed(2)); // Refund 50-100% of sale
    const goodeatsFeesRefunded = parseFloat((refundAmount * 0.05).toFixed(2)); // Assume 5% fee
    
    sheet4Data.push({
      'Sale ID': saleId,
      'Sale Date': formatDate(saleDate),
      'Sale Time': formatTime(saleDate),
      'Order Status': 'REFUNDED',
      'Refund Date': formatDate(refundDate),
      'Refund Amount': refundAmount,
      'Goodeats Fee Refunded': goodeatsFeesRefunded,
      'Reference Number': 'R' + Math.random().toString(36).substring(2, 11).toUpperCase(),
      'Reason': ['Customer dissatisfied', 'Wrong order', 'Quality issue', 'Changed mind'][Math.floor(Math.random() * 4)],
      'Source': ['In-store', 'Phone', 'Email', 'App'][Math.floor(Math.random() * 4)]
    });
  }
}

// Create sheets (in an actual application, this would create an Excel file)
// The above code would generate all the data needed for your Excel file

// For this demonstration, here's a sample of the data structure for each sheet:

// Sheet 1 (Main Transactions) - First 2 rows preview:
// [
//   {
//     "Sale ID": "73f9ab3c-e6d2-4c83-885d-85e92a6b8c7f",
//     "Sale Date": "2025-02-22",
//     "Sale Time": "17:42:13",
//     "Sale Type": "INSTORE",
//     "Order Status": "COMPLETED",
//     ...
//   },
//   {
//     "Sale ID": "52a1d7c8-f304-4e0a-ab61-9f3bc7ef4d5e",
//     "Sale Date": "2025-02-22",
//     "Sale Time": "19:15:30",
//     "Sale Type": "INSTORE",
//     "Order Status": "COMPLETED",
//     ...
//   }
// ]

// The data is ready to be exported to Excel. To download it, you would need to:
// 1. Convert the JSON data to a workbook
// 2. Write the workbook to a file
// 3. Download the file

console.log("Data generation complete. Ready to export to Excel.");
console.log(`Generated ${sheet1Data.length} transactions across 3 days (${startDate.toISOString().split('T')[0]} to ${new Date(startDate.getTime() + 2 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]})`);
console.log(`All transactions have times after 15:40:00 as requested.`);
console.log(`Sheet 1 (Main Transactions): ${sheet1Data.length} records`);
console.log(`Sheet 2 (Line Items): ${sheet2Data.length} records`);
console.log(`Sheet 3 (Payment Details): ${sheet3Data.length} records`);
console.log(`Sheet 4 (Refunds): ${sheet4Data.length} records`);
console.log(`Sheet 5 (Additional Product Details): ${sheet5Data.length} records`);

// Fix the exportToExcel function
function exportToExcel() {
  try {
    // Create a new workbook
    const wb = utils.book_new();
    
    // Convert each dataset to worksheet and add to workbook
    const ws1 = utils.json_to_sheet(sheet1Data);
    const ws2 = utils.json_to_sheet(sheet2Data);
    const ws3 = utils.json_to_sheet(sheet3Data);
    const ws4 = utils.json_to_sheet(sheet4Data);
    const ws5 = utils.json_to_sheet(sheet5Data);
    
    // Add worksheets to workbook
    utils.book_append_sheet(wb, ws1, "Sales");
    utils.book_append_sheet(wb, ws2, "Sales Items");
    utils.book_append_sheet(wb, ws3, "Sales Payments");
    utils.book_append_sheet(wb, ws4, "Sales Refunds");
    utils.book_append_sheet(wb, ws5, "Deleted Sales Items");
    
    // Use write function directly, not as a method of utils
    // This should fix the "utils.write is not a function" error
    const buffer = write(wb, { type: 'buffer', bookType: 'xlsx' });
    fs.writeFileSync('bakery_sales_data.xlsx', buffer);
    
    console.log("Excel file 'bakery_sales_data.xlsx' has been created successfully!");
  } catch (error) {
    console.error("Error creating Excel file:", error);
  }
}

// Call the export function
exportToExcel();