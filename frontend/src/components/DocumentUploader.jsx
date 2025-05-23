import { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Button,
  Flex,
  Heading,
  Text,
  VStack,
  HStack,
  useToast,
  Progress,
  Badge,
  Icon,
  List,
  ListItem,
  useColorModeValue,
  Spinner,
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
  useDisclosure,
} from '@chakra-ui/react';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiFile, FiCheckCircle, FiXCircle, FiTrash2 } from 'react-icons/fi';
import axios from 'axios';

/**
 * Component for uploading documents and displaying previously uploaded documents.
 * Integrates with a backend API for upload, listing, and deletion.
 * Uses Chakra UI for styling and react-dropzone for drag-and-drop functionality.
 */
const DocumentUploader = ({ onDocumentsUploaded }) => {
  // --- State Management ---
  const [files, setFiles] = useState([]); // Files selected for upload
  const [uploading, setUploading] = useState(false); // Upload in progress state
  const [uploadProgress, setUploadProgress] = useState(0); // Overall upload progress (for multiple files)
  const [uploadedFiles, setUploadedFiles] = useState([]); // List of documents already uploaded and processed
  const [loadingDocuments, setLoadingDocuments] = useState(true); // State for initially loading existing documents
  const [deleteId, setDeleteId] = useState(null); // ID of the document to be deleted (for confirmation dialog)

  // --- Chakra UI Hooks ---
  const { isOpen, onOpen, onClose } = useDisclosure(); // Hook for AlertDialog state
  const toast = useToast(); // Hook for displaying notifications

  // --- Color Mode Hooks for Styling ---
  const borderColor = useColorModeValue('blue.300', 'blue.500');
  const hoverBg = useColorModeValue('blue.50', 'blue.900');
  const activeBg = useColorModeValue('blue.100', 'blue.800');

  // --- Effects ---
  // Fetch existing documents on component mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  // --- API Interaction Functions ---

  /**
   * Fetches the list of uploaded documents from the backend.
   * Calls GET /api/documents/
   * Endpoint path based on OpenAPI spec and backend router prefixing.
   */
  const fetchDocuments = async () => {
    setLoadingDocuments(true);
    try {
      // Using /api/documents/ based on OpenAPI spec and backend router prefix
      const response = await axios.get('/api/documents/');

      // Check if documents exist and is an array
      if (response.data && Array.isArray(response.data.documents)) {
        // Assuming the backend returns documents with 'id', 'filename', 'size'
        // Add 'status' and 'chunks' for frontend display, defaulting chunks to 'N/A'
        setUploadedFiles(response.data.documents.map(doc => ({
          ...doc,
          // Assume documents fetched are processed, unless backend provides status
          status: doc.status || 'processed',
          // Use actual chunks if available from the list endpoint, else N/A
          chunks: doc.chunks || 'N/A',
          // Ensure 'id' exists, using filename as fallback if backend doesn't provide a unique ID
          id: doc.id || doc.filename // Use backend ID if available, otherwise filename
        })));
      } else {
        // If no documents or invalid response, set empty array
        setUploadedFiles([]);
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
      // Always set to empty array on error fetching documents to avoid stale data
      setUploadedFiles([]);

      // Show error toast only if not initial loading and not a 404 (which means no documents yet)
      // Use optional chaining (?.) for safe access to error.response
      if (!loadingDocuments && error.response?.status !== 404) {
         // Check if it's a network error (backend likely offline)
         if (!error.response) {
             console.log("Backend appears to be offline, silently setting empty documents");
         } else {
           toast({
             title: 'Error fetching documents',
             description: 'Could not load your documents. The server might be starting up.',
             status: 'warning',
             duration: 3000,
             isClosable: true,
           });
         }
      }
    } finally {
      setLoadingDocuments(false);
    }
  };

  /**
   * Handles file drop/selection, filters supported types, and adds to state.
   * Memoized using useCallback.
   */
  const onDrop = useCallback((acceptedFiles) => {
    // Filter for supported file types based on MIME type and extensions
    const supportedFiles = acceptedFiles.filter(
      (file) =>
        file.type === 'text/plain' ||
        file.type === 'application/pdf' ||
        file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' || // .docx
        file.type === 'application/msword' || // .doc
        file.type === 'application/vnd.openxmlformats-officedocument.presentationml.presentation' || // .pptx
        file.type === 'application/vnd.ms-powerpoint' // .ppt
    );

    // Notify user if unsupported files were filtered out
    if (supportedFiles.length < acceptedFiles.length) {
      toast({
        title: 'Unsupported file type',
        description: 'Only .txt, .pdf, .docx, .doc, .pptx, and .ppt files are currently supported.',
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
    }

    // Add supported files to the list of files to be uploaded
    setFiles((prevFiles) => [...prevFiles, ...supportedFiles]);
  }, [toast]); // Dependency array includes toast

  // --- React Dropzone Hook ---
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, // Assign the onDrop handler
    accept: { // Define accepted file types for filtering
      'text/plain': ['.txt'],
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'application/vnd.ms-powerpoint': ['.ppt'],
    }
  });

  /**
   * Uploads selected files to the backend one by one.
   * Tracks progress and provides feedback via toasts.
   * Calls POST /api/documents/upload
   * Endpoint path based on OpenAPI spec and backend router prefixing.
   *
   * NOTE: The ECONNRESET error on upload might be happening within the backend's
   * handling of the background task (process_document) if it crashes or
   * takes too long without proper error handling.
   */
  const uploadFiles = async () => {
    if (files.length === 0) return; // Do nothing if no files are selected

    setUploading(true);
    setUploadProgress(0);

    const uploadResults = [];
    let successCount = 0;
    let failCount = 0;

    try {
      // Iterate and upload each file individually
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append('file', file); // Append the file to FormData

        // Calculate base progress before starting the current file's upload
        const fileProgressBase = (i / files.length) * 100;
        setUploadProgress(fileProgressBase);

        // Set a timeout for the request to prevent indefinite hanging
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
          controller.abort(); // Abort the request after timeout
          console.warn(`Upload timeout (${120}s) for file: ${file.name}`);
          // Manually add a failed result if timeout occurs
          uploadResults.push({
            filename: file.name,
            status: 'error',
            message: 'Upload timed out'
          });
          failCount++;
          // You might want to update state or show a toast here immediately for the timed-out file
        }, 120000); // 2 minute timeout

        try {
          // Make the POST request to the upload endpoint
          // Using /api/documents/upload based on OpenAPI spec and backend router prefix
          const response = await axios.post('/api/documents/upload', formData, {
            headers: {
              'Content-Type': 'multipart/form-data', // Important header for file uploads
            },
            signal: controller.signal, // Link abort controller to the request signal
            onUploadProgress: (progressEvent) => {
              // Calculate overall progress considering current file's upload progress
              const fileWeight = 100 / files.length;
              const currentFileProgress = (progressEvent.loaded / progressEvent.total) * fileWeight;
              const overallProgress = fileProgressBase + currentFileProgress;
              setUploadProgress(Math.min(overallProgress, 99)); // Cap at 99% before final step
            }
          });

          clearTimeout(timeoutId); // Clear timeout if request finishes successfully

          // Process successful upload response (status 202)
          // The backend returns file info including a simple ID (filename)
          successCount++;
          uploadResults.push({
            filename: file.name,
            status: 'processing', // Backend returns 'processing' status
            size: file.size,
            ...response.data.file // Include file info from backend response
          });

        } catch (fileError) {
          clearTimeout(timeoutId); // Clear timeout even on file error

          // Process failed upload for the current file
          failCount++;
          console.error(`Upload failed for file ${file.name}:`, fileError);
          // Use optional chaining (?.) for safe access to error.response.data.detail
          uploadResults.push({
            filename: file.name,
            status: 'error',
            message: fileError.response?.data?.detail || fileError.message || 'Upload failed'
          });
        }
      }

      // Set final progress to 100% after all files are attempted
      setUploadProgress(100);

      // Show summary toasts based on overall results
      if (successCount > 0) {
        toast({
          title: `${successCount} document${successCount > 1 ? 's' : ''} uploaded successfully`,
          description: "Processing started in the background.", // Indicate background processing
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
      }

      if (failCount > 0) {
        toast({
          title: `${failCount} document${failCount > 1 ? 's' : ''} failed to upload`,
          description: "Check console for details or try smaller files.", // More helpful error message
          status: 'error',
          duration: 7000, // Longer duration for error summary
          isClosable: true,
        });
      }

      // Refresh the list of uploaded documents from the server to reflect changes
      // This will show the newly uploaded files with their 'processing' status initially
      await fetchDocuments();

      // Notify parent component with the list of successfully *initiated* uploads
      // The parent might want to track these as 'processing' until fetchDocuments updates their status
      if (onDocumentsUploaded && typeof onDocumentsUploaded === 'function') {
        onDocumentsUploaded(uploadResults.filter(doc => doc.status !== 'error'));
      }

      // Clear the list of selected files after the upload attempt
      setFiles([]);

    } catch (error) {
      // This catch block handles errors *before* the per-file loop starts,
      // or unexpected errors during the overall process setup.
      console.error('Overall upload process error:', error);
      toast({
        title: 'Upload process error',
        description: 'An unexpected error occurred during the upload process.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setUploading(false);
      // Reset progress after a short delay to show 100% briefly
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  /**
   * Removes a file from the list of files selected for upload (before uploading).
   * @param {number} index - The index of the file to remove.
   */
  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  /**
   * Sets the ID of the document to be deleted and opens the confirmation dialog.
   * @param {string} id - The ID of the document to delete.
   */
  const confirmDelete = (id) => {
    setDeleteId(id); // Store the ID of the document to delete
    onOpen(); // Open the AlertDialog
  };

  /**
   * Deletes the document using the stored deleteId after confirmation.
   * Calls DELETE /api/documents/{filename}
   * Endpoint path based on OpenAPI spec and backend router prefixing.
   *
   * NOTE: This function assumes your backend's DELETE endpoint uses the
   * filename as a path parameter, not the document ID. It finds the filename
   * from the `uploadedFiles` list using the stored `deleteId`.
   * The ECONNRESET on delete is likely happening in the backend's delete
   * endpoint implementation or related document/embedding cleanup.
   */
  const deleteDocument = async () => {
    if (!deleteId) return; // Should not happen if dialog is open, but safety check

    // Find the document to delete by ID from the currently displayed list
    const documentToDelete = uploadedFiles.find(file => file.id === deleteId);

    // Check if the document was found and has a filename property
    if (!documentToDelete || !documentToDelete.filename) {
       console.error("Attempted to delete document with unknown ID or missing filename:", deleteId, documentToDelete);
       toast({
         title: 'Error deleting document',
         description: 'Could not find document information.',
         status: 'error',
         duration: 5000,
         isClosable: true,
       });
       onClose(); // Close dialog on error
       setDeleteId(null); // Clear stored ID
       return; // Stop execution
    }

    try {
      // Make the DELETE request using the filename
      // Using /api/documents/{filename} based on OpenAPI spec and backend router prefix
      // Use encodeURIComponent to handle special characters in filenames
      await axios.delete(`/api/documents/${encodeURIComponent(documentToDelete.filename)}`);

      // Optimistically update the UI by removing the deleted document
      setUploadedFiles(prev => prev.filter(file => file.id !== deleteId));

      toast({
        title: 'Document deleted',
        description: 'The document has been successfully deleted.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });

      // Re-fetch the list to ensure state is fully in sync with the backend
      // This is important if backend cleanup is asynchronous or has side effects
      fetchDocuments();

    } catch (error) {
      console.error('Error deleting document:', error);
      // Use optional chaining (?.) for safe access to error.response.data.detail
      toast({
        title: 'Error deleting document',
        description: error.response?.data?.detail || 'An error occurred while deleting the document.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      // Re-fetch to sync state after failure in case optimistic update failed or backend state is inconsistent
      fetchDocuments();
    } finally {
      onClose(); // Always close the dialog
      setDeleteId(null); // Always clear the stored ID
    }
  };


  // --- Rendered Component Structure ---
  return (
    <VStack spacing={6} align="stretch" w="100%"> {/* Added w="100%" for consistency */}
      {/* --- File Dropzone Area --- */}
      <Box
        {...getRootProps()} // Spread react-dropzone root props
        p={6} // Adjusted padding slightly for better spacing
        borderWidth={2}
        borderRadius="lg"
        borderStyle="dashed"
        borderColor={isDragActive ? "blue.500" : borderColor} // Dynamic border color based on drag state
        bg={isDragActive ? activeBg : 'transparent'} // Dynamic background color
        _hover={{ bg: hoverBg }} // Hover effect using color mode hook
        cursor="pointer"
        transition="all 0.2s"
        // Removed duplicate _hover and _active props - _hover already defines the style
      >
        <input {...getInputProps()} /> {/* Spread react-dropzone input props */}
        <VStack spacing={2}>
          <Icon as={FiUpload} boxSize={10} color="blue.500" /> {/* Upload icon, used boxSize */}
          <Heading size="md">
            {isDragActive ? 'Drop files here' : 'Drag & drop files or click to browse'} {/* Dynamic text */}
          </Heading>
          <Text color="gray.500">Supported formats: .txt, .pdf, .docx, .doc, .pptx, .ppt</Text> {/* Supported formats info */}
        </VStack>
      </Box>

      {/* --- Selected Files List (to be uploaded) --- */}
      {files.length > 0 && ( // Only show if there are files selected for upload
        <Box>
          <Heading size="sm" mb={3}> {/* Adjusted margin-bottom */}
            Selected Files ({files.length})
          </Heading>
          <List spacing={2}> {/* Adjusted spacing */}
            {files.map((file, index) => (
              <ListItem key={index} p={2} borderWidth={1} borderRadius="md"> {/* Moved padding, border, radius to ListItem */}
                <Flex justify="space-between" align="center">
                  <HStack>
                    <Icon as={FiFile} color="blue.500" /> {/* File icon */}
                    <Text fontWeight="medium" isTruncated maxW="300px">{file.name}</Text> {/* Added fontWeight, truncation for long names */}
                    <Text fontSize="xs" color="gray.500">{(file.size / 1024).toFixed(1)} KB</Text> {/* Consistent size display */}
                  </HStack>
                  <Button
                    size="sm"
                    colorScheme="red"
                    variant="ghost" // Use ghost variant for less visual weight
                    onClick={() => removeFile(index)} // Remove button handler
                    leftIcon={<FiTrash2 />} // Added trash icon
                  >
                    Remove
                  </Button>
                </Flex>
              </ListItem>
            ))}
          </List>

          {/* Upload Button */}
          <Button
            mt={4} // Margin top
            colorScheme="blue"
            isLoading={uploading} // Show loading state based on 'uploading' state
            loadingText={`Uploading... ${uploadProgress.toFixed(0)}%`} // Improved loading text with progress
            onClick={uploadFiles} // Upload handler
            isDisabled={files.length === 0 || uploading} // Disable if no files or already uploading
            width="full" // Make button full width
          >
            Upload {files.length} {files.length === 1 ? "file" : "files"} {/* Dynamic button text */}
          </Button>

          {/* Upload Progress Bar */}
          {uploading && <Progress value={uploadProgress} mt={2} size="sm" colorScheme="blue" borderRadius="full" />} {/* Added size and borderRadius */}
        </Box>
      )}

      {/* --- Uploaded Documents List --- */}
      <Box mt={6}> {/* Adjusted margin-top */}
        <Heading size="md" mb={4}> {/* Adjusted margin-bottom */}
          Uploaded Documents
        </Heading>
        {/* Loading Spinner for fetching existing documents */}
        {loadingDocuments ? (
          <Flex justify="center" py={8}> {/* Adjusted padding-y */}
            <Spinner size="xl" color="blue.500" /> {/* Used size="xl" for a larger spinner */}
          </Flex>
        ) : uploadedFiles.length === 0 ? (
          // Empty state message when no documents are uploaded
          <Box p={6} borderWidth={1} borderRadius="lg" textAlign="center"> {/* Adjusted padding, border, radius */}
            <Text color="gray.500">No documents uploaded yet. Upload documents to start asking questions.</Text> {/* Improved empty state message */}
          </Box>
        ) : (
          // List of uploaded documents fetched from the backend
          <List spacing={3}> {/* Adjusted spacing */}
            {uploadedFiles.map((file) => ( // Using file.id as the key, assuming it's unique and stable
              <ListItem key={file.id} p={3} borderWidth={1} borderRadius="md" boxShadow="sm"> {/* Added box shadow */}
                <Flex justify="space-between" align="center">
                  <HStack spacing={3}> {/* Adjusted spacing */}
                    {/* Status Icon (Check for processed, X for error - based on frontend state/assumption) */}
                    <Icon
                      as={file.status === 'processed' ? FiCheckCircle : FiXCircle}
                      color={file.status === 'processed' ? 'green.500' : 'red.500'}
                      boxSize={5} // Consistent icon size
                    />
                    <Box>
                       <Text fontWeight="medium">{file.filename}</Text> {/* Added fontWeight */}
                       <HStack spacing={2} mt={1}> {/* Spacing for badges/size */}
                         {/* Status Badge */}
                         <Badge colorScheme={file.status === 'processed' ? 'green' : 'red'}>
                           {file.status}
                         </Badge>
                         {/* Chunks Info Badge (if available and not 'N/A') */}
                         {file.chunks && file.chunks !== 'N/A' && (
                           <Badge colorScheme="blue">{file.chunks} chunks</Badge>
                         )}
                         {/* File Size Display */}
                         {file.size !== undefined && file.size !== null && ( // Ensure size exists before displaying
                            <Text fontSize="xs" color="gray.500">
                              {typeof file.size === "number" // Check if size is a number to format
                               ? `${(file.size / 1024).toFixed(1)} KB`
                               : file.size // Display as is if not a number (e.g., "N/A" string)
                              }
                           </Text>
                         )}
                       </HStack>
                    </Box>
                  </HStack>
                  {/* Delete Button */}
                  <Button
                    size="sm"
                    colorScheme="red"
                    variant="ghost" // Use ghost variant
                    leftIcon={<FiTrash2 />} // Trash icon
                    onClick={() => confirmDelete(file.id)} // Call confirmDelete with document ID
                  >
                    Delete {/* Button text */}
                  </Button>
                </Flex>
              </ListItem>
            ))}
          </List>
        )}
      </Box>

      {/* --- Delete Confirmation AlertDialog --- */}
      <AlertDialog isOpen={isOpen} onClose={onClose} leastDestructiveRef={undefined}>
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Delete Document
            </AlertDialogHeader>

            <AlertDialogBody>
              Are you sure you want to delete this document? This action cannot be undone.
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button onClick={onClose}>
                Cancel
              </Button>
              <Button colorScheme="red" onClick={deleteDocument} ml={3}>
                Delete
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </VStack>
  );
};

export default DocumentUploader;
