import { useState, useRef } from 'react';
import {
  Box,
  Button,
  Flex,
  FormControl,
  Heading,
  Input,
  Text,
  VStack,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Spinner,
  useColorModeValue,
  Alert,
  AlertIcon,
  InputGroup,
  InputRightElement,
  IconButton,
} from '@chakra-ui/react';
import { FiSend } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
// We will use fetch for non-streaming, so axios is no longer strictly needed for the query
// import axios from 'axios';

/**
 * Frontend component for the query interface.
 * Handles user input, sends queries to the backend, displays answers and sources,
 * and manages query history. Now expects a single, non-streaming response.
 */
const QueryInterface = ({ uploadedDocs = [] }) => {
  // --- State Management ---
  const [query, setQuery] = useState(''); // Current query input
  const [answer, setAnswer] = useState(''); // The answer received from the backend
  const [sources, setSources] = useState([]); // Source documents/chunks for the answer
  const [isLoading, setIsLoading] = useState(false); // Loading state for query
  const [error, setError] = useState(''); // Error message state
  const [queryHistory, setQueryHistory] = useState([]); // Array to store past queries and responses
  // Removed isStreaming state as we are no longer streaming in the frontend

  // Ref for the input element to allow focusing
  const inputRef = useRef(null);

  // --- Color Mode Hooks for Styling ---
  const bgColor = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const hoverBgColor = useColorModeValue('gray.50', 'gray.600');

  /**
   * Handles the submission of the query form.
   * Sends the query to the backend API and processes the complete response.
   */
  const handleSubmit = async (e) => {
    // Prevent default form submission if triggered by form event
    e?.preventDefault();
    // Do nothing if query is empty or only whitespace
    if (!query.trim()) return;

    // Capture the current query for history before clearing the input
    const currentQuery = query;

    // Reset states and set loading
    setIsLoading(true);
    setError('');
    setAnswer(''); // Clear previous answer
    setSources([]); // Clear previous sources

    // Clear the input field immediately after capturing the query
    setQuery('');

    try {
      // Check if documents have been uploaded before querying
      if (uploadedDocs.length === 0) {
        const noDocsMessage = "I don't have any documents to search through. Please upload some documents first.";
        setAnswer(noDocsMessage);
        setSources([]);

        // Add to history even for no-document queries
        setQueryHistory((prev) => [
          {
            query: currentQuery,
            answer: noDocsMessage,
            sources: [],
            timestamp: new Date().toISOString()
          },
          ...prev,
        ]);
        setIsLoading(false); // Stop loading
        return; // Exit the function
      }

      // Create JSON payload for the backend request
      const payload = {
        query: currentQuery, // Use the captured query
        max_tokens: 1024,
        temperature: 0.7,
        detailed: true,
        fast_mode: false // Example parameter, adjust as needed
        // documents: selectedDocumentIds // Include if filtering is implemented
      };

      // --- Use fetch API for non-streaming ---
      const response = await fetch('/api/query/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      // Check for non-OK HTTP status codes
      if (!response.ok) {
        const errorData = await response.json(); // Attempt to read error details
        throw new Error(`Backend Error: ${response.status} ${response.statusText} - ${errorData?.detail || 'Unknown error'}`);
      }

      // Parse the complete JSON response
      const result = await response.json();

      // Process the successful response from the backend
      const finalAnswer = result.answer || "No answer found.";
      const receivedSources = result.sources || [];
      const finalProcessingTime = result.processing_time;
      const isComplexQuery = result.is_complex || false;


      setAnswer(finalAnswer);
      setSources(receivedSources);

      // Add the query and response to history
      setQueryHistory((prev) => [
        {
          query: currentQuery, // Use the captured query
          answer: finalAnswer, // Use the accumulated text
          sources: receivedSources, // Use the collected sources
          timestamp: new Date().toISOString(),
          processing_time: finalProcessingTime,
          is_complex: isComplexQuery
        },
        ...prev, // Add new query to the top
      ]);

    } catch (error) {
      // Handle errors that occur during the fetch request itself (e.g., network issues, initial HTTP errors)
      console.error('Overall Query Submission Error:', error);

      // Set a user-friendly error message
      setError(error.message || 'An unknown error occurred while submitting the query.');
      setAnswer("I'm sorry, I encountered an error while processing your query. Please try again.");
      setSources([]); // Clear sources on error

       // Add the failed query to history with error information
       setQueryHistory((prev) => [
           {
               query: currentQuery, // Use the captured query
               answer: `Error: ${error.message || 'Unknown error'}`, // Indicate error in answer
               sources: [],
               timestamp: new Date().toISOString(),
               processing_time: 0, // Or indicate error time
               is_complex: false, // Or unknown
               error: error.message // Store error detail
           },
           ...prev,
       ]);

    } finally {
      // Always set loading to false when the process finishes
      setIsLoading(false);
      // Removed setIsStreaming(false)
      // Optionally re-focus the input after processing
      if (inputRef.current) {
        inputRef.current.focus();
      }
    }
  };

  /**
   * Handles clicking on a history item to load it back into the query interface.
   * @param {object} historyItem - The history item object containing query, answer, sources.
   */
  const handleHistoryItemClick = (historyItem) => {
    // Populate the current query, answer, and sources with the history item's data
    setQuery(historyItem.query);
    setAnswer(historyItem.answer);
    setSources(historyItem.sources);
    setError(historyItem.error || ''); // Load error if present
    // Note: isStreaming and isLoading should be false when loading from history

    // Focus the input field after loading history
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  // --- Rendered Component Structure ---
  return (
    <Flex direction={{ base: 'column', md: 'row' }} gap={6}>
      {/* --- Main Query Area --- */}
      <VStack flex="1" spacing={6} align="stretch">
        {/* Query Input Form */}
        <form onSubmit={handleSubmit}>
          <FormControl>
            <InputGroup size="lg">
              <Input
                ref={inputRef} // Attach ref to input
                placeholder={isLoading ? "Processing answer..." : "Ask a question about your documents..."} // Update placeholder
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                pr="4.5rem" // Space for the send button
                onKeyPress={(e) => { // Allow submitting with Enter key
                    if (e.key === 'Enter' && !e.shiftKey) { // Check for Enter key without Shift
                        handleSubmit(); // Call submit handler
                        e.preventDefault(); // Prevent default form submission
                    }
                }}
                isDisabled={isLoading} // Disable input while loading
              />
              <InputRightElement width="4.5rem">
                <IconButton
                  h="1.75rem" // Height
                  size="sm" // Size
                  colorScheme="blue"
                  icon={<FiSend />} // Send icon
                  isLoading={isLoading} // Use isLoading for button spinner
                  type="submit" // Make it a submit button
                  aria-label="Send query"
                  isDisabled={!query.trim() || isLoading} // Disable if query is empty or loading
                />
              </InputRightElement>
            </InputGroup>
          </FormControl>
        </form>

        {/* Info Alert: No documents uploaded */}
        {uploadedDocs.length === 0 && !answer && !isLoading && !error && ( // Show only initially if no docs and no query/error
          <Alert status="info" borderRadius="md">
            <AlertIcon />
            Please upload some documents first to start asking questions.
          </Alert>
        )}

        {/* Error Alert */}
        {error && (
          <Alert status="error" borderRadius="md">
            <AlertIcon />
            {error}
          </Alert>
        )}

        {/* Loading Spinner (Show while loading) */}
        {isLoading && (
          <Box textAlign="center" py={10}>
            <Spinner size="xl" color="blue.500" />
            <Text mt={4}>Processing your query...</Text> {/* Update loading text */}
          </Box>
        )}

        {/* Answer and Sources Display */}
        {answer && !isLoading && ( // Show answer if available and not loading
          <Box p={6} borderWidth={1} borderRadius="lg" bg={bgColor} boxShadow="md">
            <Heading size="md" mb={4}>
              Answer
            </Heading>
            {/* Use ReactMarkdown to render the answer */}
            <Box className="markdown-content">
              {/* Display the answer state */}
              <ReactMarkdown>{answer}</ReactMarkdown>
            </Box>

            {/* Sources Accordion */}
            {sources.length > 0 && (
              <Box mt={6}>
                <Heading size="sm" mb={2}>
                  Sources
                </Heading>
                <Accordion allowMultiple> {/* Allow multiple source panels to be open */}
                  {sources.map((source, index) => (
                    // Use source ID or a combination if available, fallback to index
                    <AccordionItem key={source.id || index} borderColor={borderColor}>
                      <h2>
                        <AccordionButton>
                          <Box flex="1" textAlign="left">
                            <Text fontWeight="medium">
                              {/* This is the line that displays the source name */}
                              Source {index + 1}: {source.metadata?.original_filename || 'Document'}
                            </Text>
                          </Box>
                          <AccordionIcon />
                        </AccordionButton>
                      </h2>
                      <AccordionPanel pb={4}>
                        {/* Display source content, preserving whitespace/line breaks */}
                        <Text whiteSpace="pre-wrap" fontSize="sm">{source.content}</Text> {/* Added fontSize */}
                      </AccordionPanel>
                    </AccordionItem>
                  ))}
                </Accordion>
              </Box>
            )}
          </Box>
        )}
      </VStack>

      {/* --- Query History Sidebar --- */}
      <Box
        width={{ base: '100%', md: '300px' }} // Responsive width
        borderWidth={1}
        borderRadius="lg"
        p={4}
        bg={bgColor}
        height="fit-content" // Adjust height based on content
        flexShrink={0} // Prevent shrinking below base width
      >
        <Heading size="md" mb={4}>
          Query History
        </Heading>
        {queryHistory.length === 0 ? (
          <Text color="gray.500">No queries yet</Text>
        ) : (
          <VStack align="stretch" spacing={2} maxH="500px" overflowY="auto"> {/* Fixed max height and overflow */}
            {/* Map through query history items */}
            {queryHistory.map((item, index) => (
              <Box
                key={index} // Using index for history items key
                p={3}
                borderWidth={1}
                borderRadius="md"
                cursor="pointer" // Indicate clickable
                _hover={{ bg: hoverBgColor }} // Hover effect
                onClick={() => handleHistoryItemClick(item)} // Click handler to load history
              >
                <Text fontWeight="medium" noOfLines={1}> {/* Show only one line of the query */}
                  {item.query}
                </Text>
                <Text fontSize="xs" color="gray.500">
                  {/* Display timestamp in a readable format */}
                  {new Date(item.timestamp).toLocaleTimeString()}
                  {item.processing_time !== undefined && ` (${item.processing_time.toFixed(2)}s)`} {/* Display processing time if available */}
                </Text>
              </Box>
            ))}
          </VStack>
        )}
      </Box>
    </Flex>
  );
};

export default QueryInterface;
