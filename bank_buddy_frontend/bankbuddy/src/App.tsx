import { useState } from 'react';
import './App.css';
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
import { MainContainer, ChatContainer, MessageList, Message, MessageInput, TypingIndicator, Avatar, MessageTextContent } from '@chatscope/chat-ui-kit-react';


function App() {

  const [messages, setMessages] = useState([
    {
      message: "Hello, I'm Bank Buddy. Your Commbank assistant. Ask me anything!",
      sentTime: "just now",
      sender: "BankBuddy"
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = async (message: string) => {
    const newMessage = {
      message: message,
      sentTime: "just now",
      sender: "user"
    };

    setMessages((prevMessages) => {
      const newMessages = [...prevMessages, newMessage];
      getResponse(message); // Pass newMessages to getResponse
      return newMessages;
    });

    setIsTyping(true);
  };

  const getResponse = (message: string) => {
    fetch('http://127.0.0.1:8000/chat/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_question: message }),
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      console.log('Response data:', data);
      const newResp = {
        message: data.answer as string,
        sentTime: "just now",
        sender: "BankBuddy"
      };
      setMessages((prevMessages) => [...prevMessages, newResp]);
      setIsTyping(false);
    })
    .catch(error => {
      console.error('Error:', error);
      setIsTyping(false);
    });
  };


  return (
    
        <div className="App">
          <div className='top-text'>
            <h4>BankBuddy</h4>
          </div>
        <div className='chat-div'>
          <MainContainer>
            <ChatContainer className='div-width'>       
              <MessageList 
                scrollBehavior="smooth" 
                typingIndicator={isTyping ? <TypingIndicator content="Thinking" /> : null}
              >
                {messages.map((message, i) => {
                  console.log(message)

                  if (message.sender === 'user') {
                    return <Message key={i}  avatarPosition={'top-right'} model={{...message, direction: 'outgoing', position: 1}}>
                      <Avatar  size='md' src='/src/assets/manvec.jpg'></Avatar>

                    </Message>
                  } else if (message.sender === 'BankBuddy') {
                  return <Message key={i}  avatarPosition={'top-left'} model={{...message, direction: 'incoming', position: 1}}>
                    <Avatar  size='md' src='/src/assets/robovec.jpg'></Avatar>
                  </Message>
                  }
                })}
              </MessageList>
              <MessageInput attachButton={false} placeholder="Ask BankBuddy" onSend={handleSend} />        

            </ChatContainer>
          </MainContainer>
        </div>

      </div>
          

  );
}
export default App;
