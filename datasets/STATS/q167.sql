select  count(*) from postHistory as ph,          votes as v,  		users as u,  		badges as b  where u.Id = ph.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND u.Reputation>=1  AND u.Reputation<=567  AND u.UpVotes>=0  AND u.UpVotes<=75  AND u.CreationDate>='2010-10-25 06:43:31'::timestamp;