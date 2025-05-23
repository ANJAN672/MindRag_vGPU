import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { 
  Box, 
  Container, 
  Flex, 
  Heading, 
  Text, 
  VStack,
  useColorMode,
  useColorModeValue,
  IconButton,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  CloseButton,
  Spinner
} from '@chakra-ui/react'
import { MoonIcon, SunIcon } from '@chakra-ui/icons'
import DocumentUploader from './components/DocumentUploader'
import QueryInterface from './components/QueryInterface'

function App() {
  const { colorMode, toggleColorMode } = useColorMode();
  const bgColor = useColorModeValue('gray.50', 'gray.800');
  const textColor = useColorModeValue('gray.800', 'white');
  const [uploadedDocs, setUploadedDocs] = useState([]);
  const [activeTab, setActiveTab] = useState(0);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking', 'online', 'offline'
  const [showStatusAlert, setShowStatusAlert] = useState(true);
  
  // Check backend health on component mount
  useEffect(() => {
    // First try to check if the backend is available
    checkBackendAvailability();
    
    // Set up interval to check health every 10 seconds, but only if initially offline
    const intervalId = setInterval(() => {
      if (backendStatus !== 'online') {
        checkBackendAvailability();
      }
    }, 10000);
    
    // Hide the offline message after 8 seconds even if backend is offline
    // This prevents confusion for users who might not have started the backend yet
    const hideAlertTimeout = setTimeout(() => {
      if (backendStatus === 'offline') {
        setShowStatusAlert(false);
      }
    }, 8000);
    
    // Clean up interval and timeout on unmount
    return () => {
      clearInterval(intervalId);
      clearTimeout(hideAlertTimeout);
    };
  }, [backendStatus]);
  
  // First check if the backend is available at all
  const checkBackendAvailability = async () => {
    try {
      // Try to access the root endpoint first, which is lighter
      const response = await axios.get('/api', { timeout: 3000 });
      if (response.status === 200) {
        // If root endpoint works, backend is available
        setBackendStatus('online');
        setShowStatusAlert(false);
        
        // Now check for documents
        fetchDocuments();
      } else {
        // Try the health endpoint as fallback
        checkBackendHealth();
      }
    } catch (error) {
      // If root fails, try health endpoint
      try {
        await checkBackendHealth();
      } catch (healthError) {
        console.error('Backend appears to be offline:', healthError);
        setBackendStatus('offline');
      }
    }
  };
  
  const fetchDocuments = async () => {
    try {
      const docsResponse = await axios.get('/api/documents');
      if (docsResponse.data && Array.isArray(docsResponse.data.documents)) {
        setUploadedDocs(docsResponse.data.documents.map(doc => ({
          ...doc,
          status: 'processed'
        })));
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
      // Don't change backend status for document fetch errors
    }
  };
  
  const checkBackendHealth = async () => {
    try {
      const response = await axios.get('/api/health', { timeout: 5000 });
      if (response.data && response.data.status === 'healthy') {
        setBackendStatus('online');
        setShowStatusAlert(false);
        
        // If we have documents in the backend, update our state
        if (response.data.has_documents && uploadedDocs.length === 0) {
          fetchDocuments();
        }
        return true;
      } else {
        setBackendStatus('offline');
        return false;
      }
    } catch (error) {
      console.error('Backend health check failed:', error);
      setBackendStatus('offline');
      throw error;
    }
  };

  const handleDocumentsUploaded = (docs) => {
    // If docs is an array, it's a new upload
    if (Array.isArray(docs)) {
      // Filter out any documents that might be duplicates
      const newDocs = docs.filter(newDoc => 
        !uploadedDocs.some(existingDoc => existingDoc.id === newDoc.id)
      );
      
      if (newDocs.length > 0) {
        setUploadedDocs((prevDocs) => [...prevDocs, ...newDocs]);
        // Switch to the query tab after successful upload
        setActiveTab(1);
      }
    } else {
      // If docs is not an array, it's the updated list after deletion
      setUploadedDocs(docs);
    }
  };

  return (
    <Box minH="100vh" bg={bgColor} color={textColor}>
      <Container maxW="container.xl" py={8}>
        <Flex justifyContent="space-between" alignItems="center" mb={8}>
          <Heading 
            size="xl" 
            bgGradient="linear(to-r, blue.400, purple.500)" 
            bgClip="text"
          >
            MindRAG
          </Heading>
          <IconButton
            icon={colorMode === 'dark' ? <SunIcon /> : <MoonIcon />}
            onClick={toggleColorMode}
            variant="ghost"
            aria-label="Toggle color mode"
          />
        </Flex>
        
        {/* Backend status alert - only show after a delay if still offline */}
        {showStatusAlert && backendStatus === 'offline' && (
          <Alert status="warning" mb={4} borderRadius="md">
            <AlertIcon />
            <Box flex="1">
              <AlertTitle mr={2}>Connecting to backend...</AlertTitle>
              <AlertDescription display="block">
                If this message persists, please make sure the backend server is running.
              </AlertDescription>
            </Box>
            <CloseButton 
              position="absolute" 
              right="8px" 
              top="8px" 
              onClick={() => setShowStatusAlert(false)} 
            />
          </Alert>
        )}
        
        {/* Loading indicator - only show briefly */}
        {backendStatus === 'checking' && (
          <Alert status="info" mb={4} borderRadius="md">
            <AlertIcon />
            <Flex align="center">
              <Spinner size="sm" mr={2} />
              <Text>Initializing application...</Text>
            </Flex>
          </Alert>
        )}

        <VStack spacing={8} align="stretch">
          <Box textAlign="center">
            <Heading as="h2" size="lg" mb={2}>
              Open-Source RAG System
            </Heading>
            <Text fontSize="lg" maxW="container.md" mx="auto" color={colorMode === 'dark' ? 'gray.400' : 'gray.600'}>
              Upload documents (.txt, .pdf, .docx, .doc, .pptx, .ppt), process them with state-of-the-art AI models, and get accurate answers to your questions.
            </Text>
          </Box>

          <Tabs 
            variant="soft-rounded" 
            colorScheme="blue" 
            size="lg" 
            index={activeTab} 
            onChange={(index) => setActiveTab(index)}
          >
            <TabList mb={4}>
              <Tab>Upload Documents</Tab>
              <Tab>Ask Questions</Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                <Box p={8} borderWidth={1} borderRadius="lg" boxShadow="md">
                  <DocumentUploader onDocumentsUploaded={handleDocumentsUploaded} />
                </Box>
              </TabPanel>
              <TabPanel>
                <Box p={8} borderWidth={1} borderRadius="lg" boxShadow="md">
                  <QueryInterface uploadedDocs={uploadedDocs} />
                </Box>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </VStack>
        
        <Box as="footer" mt={12} textAlign="center" fontSize="sm" color={colorMode === 'dark' ? 'gray.500' : 'gray.600'}>
          <Text>MindRAG - Powered by MindCore Labs</Text>
        </Box>
      </Container>
    </Box>
  )
}

export default App;